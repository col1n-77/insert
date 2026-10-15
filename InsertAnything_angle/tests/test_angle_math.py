"""CPU checks independent of Isaac/Omniverse; run with `python -m unittest ...`."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch

_SPEC = importlib.util.spec_from_file_location(
    "angle_math", Path(__file__).parents[1] / "source/insertanything_angle/math.py"
)
angle_math = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(angle_math)


def rotation(axis, deg):
    a = math.radians(deg) / 2
    return torch.tensor([math.cos(a), *(math.sin(a) * x for x in axis)], dtype=torch.float64)


def product(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2))


class TestAngleGeometry(unittest.TestCase):
    def test_signed_in_hand_tilt_and_reward(self):
        tcp = rotation((1, 0, 0), 180)
        hole = rotation((0, 0, 1), 0)
        for theta in (-10.0, 0.0, 10.0):
            obj = product(product(tcp, rotation((0, 1, 0), theta)), tcp)
            g = angle_math.angle_geometry(obj[None], tcp[None], hole[None])
            self.assertAlmostEqual(math.degrees(g["true_in_hand_angle_rad"].item()), theta, places=8)
            self.assertAlmostEqual(math.degrees(g["axis_error_rad"].item()), abs(theta), places=8)
            self.assertLess(g["off_axis_residual_rad"].item(), 1e-10)
            reward = angle_math.gaussian_axis_reward(g["axis_error_rad"], 1.0, 5.0)
            self.assertAlmostEqual(reward.item(), math.expm1(-(theta / 5)**2), places=10)

    def test_centered_penalty_bounds_weighting_and_legacy_shift(self):
        angles = torch.deg2rad(torch.tensor([0., 1., 5., 10., 30., 180.], dtype=torch.float64))
        weight = 2.5
        penalty = angle_math.gaussian_axis_reward(angles, weight, 5.)
        self.assertEqual(penalty[0].item(), 0.)
        self.assertTrue(torch.all(penalty <= 0.) and torch.all(penalty >= -weight))
        self.assertTrue(torch.all(penalty[1:] <= penalty[:-1]))
        torch.testing.assert_close(penalty, angle_math.gaussian_axis_reward(angles, 1., 5.) * weight)
        legacy = angle_math.gaussian_axis_reward(angles, weight, 5., zero_at_alignment=False)
        torch.testing.assert_close(legacy - penalty, torch.full_like(angles, weight), atol=1e-14, rtol=0)

    def test_global_rotation_invariance(self):
        tcp = rotation((1, 0, 0), 180)
        obj = product(product(tcp, rotation((0, 1, 0), -7)), tcp)
        hole = rotation((0, 0, 1), 0)
        global_q = rotation((0, 0, 1), 43)
        before = angle_math.angle_geometry(obj[None], tcp[None], hole[None])
        after = angle_math.angle_geometry(product(global_q,obj)[None], product(global_q,tcp)[None], product(global_q,hole)[None])
        for field in ("true_in_hand_angle_rad", "axis_error_rad", "off_axis_residual_rad"):
            torch.testing.assert_close(before[field], after[field], atol=1e-10, rtol=0)

    def test_compensation_preserves_in_hand_angle(self):
        # Object upright, wrist +7 about its local Y: in-hand offset is -7.
        hole = rotation((0, 0, 1), 0)
        tcp = product(rotation((1,0,0),180), rotation((0,1,0),7))
        g = angle_math.angle_geometry(hole[None], tcp[None], hole[None])
        self.assertAlmostEqual(math.degrees(g["true_in_hand_angle_rad"].item()), -7.0, places=8)
        self.assertEqual(g["axis_error_rad"].item(), 0.0)

    def test_off_axis_and_directed_axis(self):
        tcp = rotation((1,0,0),180)
        hole = rotation((0,0,1),0)
        obj = rotation((1,0,0),12)
        g = angle_math.angle_geometry(obj[None], tcp[None], hole[None])
        self.assertAlmostEqual(math.degrees(g["off_axis_residual_rad"].item()), 12, places=8)
        reverse = angle_math.angle_geometry(tcp[None], tcp[None], hole[None])
        self.assertAlmostEqual(reverse["axis_error_rad"].item(), math.pi, places=8)
        self.assertEqual(angle_math.gaussian_axis_reward(g["axis_error_rad"],0,5).item(),0)


class TestAngleNoise(unittest.TestCase):
    def test_independent_global_rng_and_repeat_read(self):
        torch.manual_seed(918)
        before = torch.get_rng_state().clone()
        cache = angle_math.AngleNoiseCache(128,"cpu",1.0,42)
        a = cache.sample().clone()
        b = cache.sample().clone()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(torch.unique(a).numel(),128)
        self.assertTrue(torch.equal(cache.sample_count, torch.ones(128,dtype=torch.long)))

    def test_partial_reset_changes_only_reset_envs(self):
        cache = angle_math.AngleNoiseCache(16,"cpu",1.0,42)
        before = cache.sample().clone()
        ids = torch.tensor([1,4,10])
        cache.invalidate(ids)
        after = cache.sample().clone()
        changed = (before != after).nonzero().flatten()
        self.assertTrue(torch.equal(ids,changed))
        self.assertTrue(torch.equal(cache.sample_count[ids],torch.full((3,),2)))
        cache.invalidate()
        self.assertTrue(torch.all(cache.sample() != after))

    def test_reseed_and_zero_noise(self):
        cache = angle_math.AngleNoiseCache(10,"cpu",1.0,93)
        first = cache.sample().clone()
        cache.invalidate()
        self.assertFalse(torch.equal(first,cache.sample()))
        cache.reseed(93)
        self.assertTrue(torch.equal(first,cache.sample()))
        clean = angle_math.AngleNoiseCache(10,"cpu",0.0,93)
        self.assertEqual(torch.count_nonzero(clean.sample()).item(),0)


class TestDepthAxisGate(unittest.TestCase):
    def test_full_reward_for_aligned_states_and_bounded_monotone_decay(self):
        angles = torch.deg2rad(torch.linspace(0, 180, 1801, dtype=torch.float64))
        gate = angle_math.axis_depth_gate(angles, 3., 5., .1)
        self.assertTrue(torch.equal(gate[:31], torch.ones(31, dtype=gate.dtype)))
        self.assertTrue(torch.all(gate[1:] <= gate[:-1]))
        self.assertTrue(torch.all(gate >= .1 - 1e-15) and torch.all(gate <= 1))
        self.assertAlmostEqual(gate[-1].item(), .1)

    def test_smooth_boundary_and_gradients(self):
        angles = torch.deg2rad(torch.tensor([2.999999, 3., 3.000001, 5., 10.], dtype=torch.float64)).requires_grad_()
        gate = angle_math.axis_depth_gate(angles, 3., 5., .1)
        gate.sum().backward()
        self.assertTrue(torch.isfinite(angles.grad).all())
        self.assertEqual(angles.grad[1].item(), 0.)
        self.assertLess(abs(angles.grad[2].item()), 1e-5)
        self.assertTrue(torch.all(angles.grad[3:] < 0))

    def test_compensated_wrist_keeps_depth_reward(self):
        hole = rotation((0, 0, 1), 0)
        tcp = product(rotation((1, 0, 0), 180), rotation((0, 1, 0), 10))
        g = angle_math.angle_geometry(hole[None], tcp[None], hole[None])
        self.assertAlmostEqual(math.degrees(g['true_in_hand_angle_rad'].item()), -10.)
        self.assertEqual(angle_math.axis_depth_gate(g['axis_error_rad'], 3., 5., .1).item(), 1.)

    def test_floor_one_is_identity_and_no_depth_term_is_noop(self):
        a = torch.deg2rad(torch.tensor([0., 3., 10., 180.]))
        self.assertTrue(torch.equal(angle_math.axis_depth_gate(a, 3., 5., 1.), torch.ones_like(a)))
        gate = angle_math.axis_depth_gate(a, 3., 5., .1)
        self.assertTrue(torch.equal((gate - 1.) * torch.zeros_like(a), torch.zeros_like(a)))

    def test_invalid_parameters_rejected(self):
        for tolerance, sigma, floor in [(-1,5,.1),(180,5,.1),(3,0,.1),(3,float('inf'),.1),
                                         (3,5,-.1),(3,5,1.1),(float('nan'),5,.1),(3,5,float('nan'))]:
            with self.subTest(tolerance=tolerance, sigma=sigma, floor=floor):
                with self.assertRaises(ValueError):
                    angle_math.axis_depth_gate(torch.zeros(2), tolerance, sigma, floor)


if __name__ == "__main__":
    unittest.main()
