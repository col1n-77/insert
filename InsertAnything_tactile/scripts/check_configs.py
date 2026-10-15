"""Check all six new configurations without stepping any task."""
import argparse
import json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import gymnasium as gym
import insertanything_tactile

rows = []
for suffix, (cfg_type, agent) in insertanything_tactile.TASKS.items():
    original_id = f"InsertAnything-{suffix}-Direct-v0"
    new_id = f"InsertAnythingTactile-{suffix}-Direct-v0"
    original_type = gym.spec(original_id).kwargs["env_cfg_entry_point"]
    before = json.dumps(original_type().to_dict(), sort_keys=True, default=str)
    cfg = cfg_type()
    after = json.dumps(original_type().to_dict(), sort_keys=True, default=str)
    assert before == after, f"Shared configuration mutation for {suffix}"
    assert cfg.robot.spawn.usd_path != original_type().robot.spawn.usd_path
    assert cfg.task.contact_force["log_contact_force"] is False
    assert cfg.scene.clone_in_fabric is False
    rows.append({"task": new_id, "configuration": cfg_type.__name__, "original_config_unchanged": True})
print("TACTILE_CONFIG_CHECK " + json.dumps(rows), flush=True)
app.close()
