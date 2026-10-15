"""InsertAnything PPO/LSTM configuration with a camera-sized batch."""
from importlib.resources import files
import yaml


def agent_cfg():
    path = files("InsertAnything.tasks.direct.insertanything.agents").joinpath("rl_games_ppo_L_cfg.yaml")
    result = yaml.safe_load(path.read_text())
    c = result["params"]["config"]
    c.update(name="InsertAnythingTactile6D", full_experiment_name=None,
             num_actors=4, horizon_length=64, minibatch_size=256, seq_length=16,
             save_frequency=25, save_best_after=1)
    c["central_value_config"]["minibatch_size"] = 256
    return result
