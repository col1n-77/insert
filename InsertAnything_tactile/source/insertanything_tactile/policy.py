"""Load an existing RL-Games policy; tactile images remain separate outputs."""
from pathlib import Path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


class CheckpointPolicy:
    def __init__(self, env, task, checkpoint, seed):
        checkpoint = Path(checkpoint).resolve(strict=True)
        cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        params = cfg["params"]
        self.device = env.unwrapped.device
        self.clip_obs = params["env"].get("clip_observations", float("inf"))
        self.clip_actions = params["env"].get("clip_actions", float("inf"))
        wrapper = RlGamesVecEnvWrapper(
            env, self.device, self.clip_obs, self.clip_actions,
            params["env"].get("obs_groups"), params["env"].get("concate_obs_groups", True),
        )
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw:
                        RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                              "env_creator": lambda **kw: wrapper})
        params["seed"] = seed
        params["load_checkpoint"] = True
        params["load_path"] = str(checkpoint)
        params["config"]["num_actors"] = env.unwrapped.num_envs
        params["config"]["device"] = self.device
        params["config"]["device_name"] = self.device
        runner = Runner()
        runner.load(cfg)
        self.agent = runner.create_player()
        self.agent.restore(str(checkpoint))
        self.agent.reset()
        self.ready = False

    def act(self, observations):
        obs = observations["policy"].clamp(-self.clip_obs, self.clip_obs).to(self.device)
        if not self.ready:
            self.agent.get_batch_size(obs, 1)
            if self.agent.is_rnn:
                self.agent.init_rnn()
            self.ready = True
        actions = self.agent.get_action(self.agent.obs_to_torch(obs), self.agent.is_deterministic)
        return actions.clamp(-self.clip_actions, self.clip_actions)

    def reset_done(self, dones):
        if self.agent.is_rnn and self.agent.states is not None:
            for state in self.agent.states:
                state[:, dones.bool(), :] = 0.0
