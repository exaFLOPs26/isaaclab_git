from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv

def sink(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
# Depends on  N_dir 
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
	sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
	sink_pos[:,2] = 0.9
	radius = 0.135

#	obj_pos = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0]], device=env.device)
#	sink_pos = torch.tensor([[0.0,0.0,10.0],[0.0,0.0,10.0]], device=env.device)
   
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
#	sorted_distances, indices = torch.sort(distance)
#	print(sorted_distances)

	# Check if the distance is within the specified radius
	result = distance <= radius
	# The result should be a single boolean value
#	result = is_near.all()
	return result

def OOB(
	env: ManagerBasedRLEnv,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("bottle0"),
	sink: SceneEntityCfg = SceneEntityCfg("sink_cabinet")
) -> bool:
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1)
# Depends on  N_dir 
	sink_pos = env.scene[sink.name].data.body_pos_w[:,0,:]
	sink_pos[:,1] = env.scene[sink.name].data.body_pos_w[:, 1, 1]
	sink_pos[:,2] = 0.9
	radius = 3.5

#	obj_pos = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0]], device=env.device)
#	sink_pos = torch.tensor([[0.0,0.0,10.0],[0.0,0.0,10.0]], device=env.device)
   
	distance = torch.linalg.norm(obj_pos - sink_pos, dim=-1)
	sorted_distances, indices = torch.sort(distance)
	print(sorted_distances)

	# Check if the distance is within the specified radius
	result = distance > radius
	# The result should be a single boolean value
#	result = is_near.all()
	return result
