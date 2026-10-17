"""Train two matched Angle policies immediately, then evaluate frozen weights.

Only axis reward weight changes (0 vs 1). Depth gating is disabled in both.
Uses unique experiment directories and checks saved configurations and sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import yaml


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key, '__MISSING__'), b.get(key, '__MISSING__'),
                                  f'{path}.{key}' if path else str(key))
        return result
    return [] if a == b else [{'field': path, 'control': a, 'aligned': b}]


def train_command(root, args, name, weight):
    return [str(root/'scripts/python.sh'), 'scripts/train_angle_rl.py', '--headless',
            '--task', args.task, '--num_envs', str(args.num_envs), '--seed', str(args.seed),
            '--max_iterations', str(args.max_iterations), '--horizon', str(args.horizon),
            '--minibatch_size', str(args.minibatch_size), '--seq_length', str(args.seq_length),
            '--save_frequency', str(args.save_frequency), '--initial_tilt_deg', str(args.initial_tilt_deg),
            '--angle_noise_std_deg', '0', '--align_sigma_deg', str(args.align_sigma_deg),
            '--align_weight', str(weight), '--learning_rate', str(args.learning_rate),
            '--lr_schedule', args.lr_schedule, '--kl_threshold', str(args.kl_threshold), '--run_name', name]


def verify_configuration(env, agent, args, weight):
    c = agent['params']['config']
    expectations = {
        'env.seed': (env['seed'], args.seed),
        'scene.num_envs': (env['scene']['num_envs'], args.num_envs),
        'initial_in_hand_tilt_deg': (env['initial_in_hand_tilt_deg'], args.initial_tilt_deg),
        'ctrl.lock_roll_pitch': (env['ctrl']['lock_roll_pitch'], False),
        'observation_space': (env['observation_space'], 27),
        'state_space': (env['state_space'], 74),
        'angle_noise_std_deg': (env['angle_obs']['noise_std_deg'], 0),
        'axis_weight': (env['axis_alignment']['reward_weight'], weight),
        'axis_sigma_deg': (env['axis_alignment']['sigma_deg'], args.align_sigma_deg),
        'axis_zero_at_alignment': (env['axis_alignment']['zero_at_alignment'], True),
        'depth_axis_gate.enabled': (env['depth_axis_gate']['enabled'], False),
        'force.enabled': (env['task']['contact_force']['enabled'], True),
        'force.use_as_reward': (env['task']['contact_force']['use_as_reward'], True),
        'training_from_scratch': (agent['params']['load_checkpoint'], False),
        'agent.seed': (agent['params']['seed'], args.seed),
        'num_actors': (c['num_actors'], args.num_envs),
        'max_epochs': (c['max_epochs'], args.max_iterations),
        'horizon_length': (c['horizon_length'], args.horizon),
        'seq_length': (c['seq_length'], args.seq_length),
        'minibatch_size': (c['minibatch_size'], args.minibatch_size),
        'mini_epochs': (c['mini_epochs'], 4),
        'learning_rate': (float(c['learning_rate']), args.learning_rate),
        'lr_schedule': (c['lr_schedule'], args.lr_schedule),
        'schedule_type': (c['schedule_type'], 'standard'),
        'kl_threshold': (float(c['kl_threshold']), args.kl_threshold),
        'critic.minibatch_size': (c['central_value_config']['minibatch_size'], args.minibatch_size),
        'critic.learning_rate': (float(c['central_value_config']['learning_rate']), args.learning_rate),
        'critic.lr_schedule': (c['central_value_config']['lr_schedule'], args.lr_schedule),
        'critic.kl_threshold': (float(c['central_value_config']['kl_threshold']), args.kl_threshold),
    }
    bad = {key: {'actual': a, 'expected': b} for key, (a, b) in expectations.items() if a != b}
    if bad:
        raise RuntimeError('Training configuration mismatch: ' + json.dumps(bad))
    return {'passed': True, 'checks': len(expectations)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study_name')
    p.add_argument('--num_envs', type=int, default=2048)
    p.add_argument('--max_iterations', type=int, default=500)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--minibatch_size', type=int, default=8192)
    p.add_argument('--seq_length', type=int, default=128)
    p.add_argument('--save_frequency', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_num_envs', type=int, default=128)
    p.add_argument('--eval_seeds', type=int, nargs='+', default=[1001, 2002])
    p.add_argument('--learning_rate', type=float, default=1.e-4)
    p.add_argument('--lr_schedule', choices=('constant', 'adaptive'), default='adaptive')
    p.add_argument('--kl_threshold', type=float, default=0.008)
    p.add_argument('--initial_tilt_deg', type=float, default=10.)
    p.add_argument('--align_sigma_deg', type=float, default=5.)
    p.add_argument('--plan_only', action='store_true', help='Save commands without launching any training')
    args = p.parse_args()
    args.task = 'InsertAnythingAngle-LHole-III-Direct-v0'
    if min(args.num_envs, args.max_iterations, args.horizon, args.minibatch_size,
           args.seq_length, args.save_frequency, args.eval_num_envs) < 1:
        p.error('All counts must be positive')
    if (args.num_envs*args.horizon % args.minibatch_size or args.horizon % args.seq_length
            or args.minibatch_size % args.seq_length):
        p.error('Batch, minibatch and horizon must have valid sequence divisibility')
    if args.seed in args.eval_seeds or len(set(args.eval_seeds)) != len(args.eval_seeds):
        p.error('Use distinct evaluation seeds, excluding the training seed')
    if not (0 <= args.initial_tilt_deg < 90 and 0 < args.align_sigma_deg < float('inf')
            and 0 < args.learning_rate < float('inf') and 0 < args.kl_threshold < float('inf')):
        p.error('Invalid tilt, sigma, learning rate or KL threshold')
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    name = args.study_name or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('LHole_III_matched_axis_w0_w1_%Y%m%d_%H%M%S')
    if Path(name).name != name:
        raise ValueError('study_name must be a directory name')
    out = root/'outputs'/name
    out.mkdir(parents=True, exist_ok=False)
    plans = {}
    for label, weight in [('control', 0.), ('aligned', 1.)]:
        run_name = f'{name}_{label}'
        run = root/'logs/rl_games/InsertAnythingAngle'/run_name
        if run.exists():
            raise FileExistsError(run)
        plans[label] = {'weight': weight, 'training_run': str(run), 'status': 'queued',
                        'command': train_command(root, args, run_name, weight), 'evaluations': {}}
    # Assert the two commands differ only in the intended weight and run name.
    commands = [plans[k]['command'].copy() for k in ('control', 'aligned')]
    for command in commands:
        command[command.index('--align_weight')+1] = '<weight>'
        command[command.index('--run_name')+1] = '<run_name>'
    assert commands[0] == commands[1]
    base = Path(os.environ.get('INSERTANYTHING_ROOT', str(root.parent)))/'source/InsertAnything/InsertAnything/tasks/direct/insertanything'
    tracked = sorted([*root.joinpath('source').rglob('*.py'), *root.joinpath('scripts').glob('*.py'),
                      root/'scripts/python.sh', *base.glob('*.py'), *base.joinpath('agents').glob('*.yaml')])
    hashes = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
    state = {'schema': 'matched_angle_reward_study_v1', 'status': 'planned' if args.plan_only else 'running',
             'started_at': now(), 'pid': os.getpid(), 'arguments': vars(args), 'arms': plans,
             'source_sha256': hashes, 'depth_axis_gate_enabled': False,
             'planned_transitions_per_arm': args.num_envs*args.horizon*args.max_iterations,
             'planned_actor_updates_per_arm': args.num_envs*args.horizon//args.minibatch_size*4*args.max_iterations,
             'comparison_note': 'Same 27/74 observations, zero added angle noise, seed, controller, force penalty, batch and LR. Only axis reward weight 0/1. One training seed; convergence and cross-seed reliability not assumed.'}
    def save(): dump(out/'study.json', state)
    save()
    print('MATCHED_STUDY_OUTPUT ' + str(out), flush=True)
    if args.plan_only:
        return
    configs = {}
    child = None
    def check_sources():
        changed = [p for p, h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
        if changed:
            raise RuntimeError('Source changed during the matched study: ' + ', '.join(changed))
    def check_configs(label):
        if label in configs:
            return
        run = Path(plans[label]['training_run'])
        paths = [run/'params/env.yaml', run/'params/agent.yaml']
        if not all(p.is_file() for p in paths):
            return
        try:
            env = yaml.load(paths[0].read_text(), Loader=yaml.FullLoader)
            agent = yaml.safe_load(paths[1].read_text())
            if not isinstance(env, dict) or not isinstance(agent, dict):
                return
            verified = verify_configuration(env, agent, args, plans[label]['weight'])
        except (yaml.YAMLError, KeyError, TypeError):
            return  # The trainer may still be writing the files.
        configs[label] = (env, agent)
        plans[label]['configuration_check'] = verified
        if all(k in configs for k in ('control', 'aligned')):
            env_diff = differences(configs['control'][0], configs['aligned'][0])
            agent_diff = differences(configs['control'][1], configs['aligned'][1])
            unexpected = [x for x in env_diff if x['field'] != 'axis_alignment.reward_weight']
            unexpected += [x for x in agent_diff if x['field'] != 'params.config.full_experiment_name']
            check = {'passed': not unexpected, 'env_differences': env_diff,
                     'agent_differences': agent_diff, 'unexpected_differences': unexpected, 'checked_at': now()}
            dump(out/'configuration_comparison.json', check)
            if unexpected:
                raise RuntimeError('Matched arms have unintended configuration differences')
        save()
        print('CONFIGURATION_VERIFIED ' + label, flush=True)
    def execute(stage, command, log, arm=None):
        nonlocal child
        check_sources()
        print('STUDY_START ' + stage + ': ' + shlex.join(command), flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            state.update(current_stage=stage, child_pid=child.pid, current_log=str(log),
                         current_progress=None, stage_started_at=now())
            save()
            while child.poll() is None:
                if arm:
                    check_configs(arm)
                with log.open('rb') as f:
                    f.seek(max(0, log.stat().st_size-65536)); tail = f.read().decode(errors='replace')
                for fatal in ('Contacts have been dropped', 'CUDA out of memory', 'Traceback (most recent call last):'):
                    if fatal in tail:
                        raise RuntimeError(f'{stage}: {fatal}; see {log}')
                epochs = re.findall(r'fps total:\s*([\d.]+).*?epoch:\s*(\d+)/(\d+)', tail)
                if epochs:
                    fps, epoch, total = epochs[-1]
                    state['current_progress'] = {'epoch': int(epoch), 'max_epochs': int(total),
                                                 'fps_total': float(fps), 'observed_at': now()}
                save()
                time.sleep(15)
            code = child.returncode
            state['child_pid'] = None
            child = None
            if code:
                raise RuntimeError(f'{stage} exited {code}; see {log}')
        if arm:
            check_configs(arm)
            if arm not in configs:
                raise RuntimeError('Training completed without verified saved configurations')
        save()
        print('STUDY_DONE ' + stage, flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for label in ('control', 'aligned'):
            arm = plans[label];run = Path(arm['training_run'])
            arm['status'] = 'training';save()
            execute('train_'+label, arm['command'], out/f'train_{label}.log', arm=label)
            contract = json.loads((run/'run.json').read_text())
            if contract['status'] != 'completed':
                raise RuntimeError('Trainer did not mark completion')
            candidates = sorted((run/'nn').glob(f'last_*_ep_{args.max_iterations}_*.pth'))
            if not candidates:
                raise RuntimeError('No final checkpoint at the planned epoch')
            selections = {'final': candidates[0]}
            best = run/'nn/InsertAnythingAngle.pth'
            if best.is_file(): selections['best_training_reward'] = best
            arm['status'] = 'evaluating';save()
            for selection, checkpoint in selections.items():
                destination = out/f'eval_{label}_{selection}'
                cmd = [str(root/'scripts/python.sh'), 'scripts/eval_angle_policy.py', '--headless',
                       '--task', args.task, '--checkpoint', str(checkpoint), '--training_run', str(run),
                       '--num_envs', str(args.eval_num_envs), '--seeds', *map(str, args.eval_seeds),
                       '--modes', 'deterministic', 'stochastic', '--output', str(destination)]
                execute(f'eval_{label}_{selection}', cmd, out/f'eval_{label}_{selection}.log')
                report = json.loads((destination/'report.json').read_text())
                if report['status'] != 'completed': raise RuntimeError('Evaluation incomplete')
                arm['evaluations'][selection] = {'report': str(destination/'report.json'), 'epoch': report['checkpoint_epoch']}
                save()
            arm['status'] = 'completed';save()
        execute('summarize', [str(root/'scripts/python.sh'), 'scripts/summarize_reward_study.py',
                             '--study', str(out)], out/'summarize.log')
        state.update(status='completed', current_stage='completed', finished_at=now());save()
        print('MATCHED_STUDY_COMPLETED ' + str(out), flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL);child.wait()
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     child_pid=None, error=repr(exc), failed_at=now());save()
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
