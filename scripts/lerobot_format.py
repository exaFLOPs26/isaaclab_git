import h5py
import numpy as np
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json

def hdf5_to_lerobot_with_videos(hdf5_path, task_json, goal_json, output_path, fps=30):
	features = {
		"kitchen_num": {"dtype": "int64", "shape": (1,), "names": None},
		"kitchen_sub_num": {"dtype": "int64", "shape": (1,), "names": None},
		"kitchen_type": {"dtype": "int64", "shape": (1,), "names": None},
		"initial_pose": {"dtype": "float32", "shape": (6,), "names": {"pose": ["x","y", "qw", "qx", "qy", "qz"]}},
		"is_first": {"dtype": "int64", "shape": (1,), "names": None},
		"is_last": {"dtype": "int64", "shape": (1,), "names": None},
		"subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
#"subtask_class": {"dtype": "string", "shape": [1], "names": None},
#		"subtask_language_instruction": {"dtype": "string", "shape": [1], "names": None},

		# Observations
		"observation.state.ee_pose": {
			"dtype": "float32",
			"shape": (22,),
			"names": {
				"motors": [
					"l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6",
					"r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6",
					"l_gripper","l_gripperR","r_gripper","r_gripperR"
				]
			}
		},

		# Images (stored as videos)
		"observation.images.front": {
			"dtype": "video",
			"shape": [480, 640, 3],
			"names": ["height", "width", "channels"],
			"info": {
				"video.fps": fps,
				"video.height": 480,
				"video.width": 640,
				"video.channels": 3,
				"video.codec": "h264",
				"video.pix_fmt": "yuv420p",
				"video.is_depth_map": False,
				"has_audio": False
			}
		},
		"observation.images.wrist_left": {
			"dtype": "video",
			"shape": [480, 640, 3],
			"names": ["height", "width", "channels"],
			"info": {
				"video.fps": fps,
				"video.height": 480,
				"video.width": 640,
				"video.channels": 3,
				"video.codec": "h264",
				"video.pix_fmt": "yuv420p",
				"video.is_depth_map": False,
				"has_audio": False
			}
		},
		"observation.images.wrist_right": {
			"dtype": "video",
			"shape": [480, 640, 3],
			"names": ["height", "width", "channels"],
			"info": {
				"video.fps": fps,
				"video.height": 480,
				"video.width": 640,
				"video.channels": 3,
				"video.codec": "h264",
				"video.pix_fmt": "yuv420p",
				"video.is_depth_map": False,
				"has_audio": False
			}
		},

		# Actions
		"action.original": {
			"dtype": "float32",
			"shape": (20,),
			"names": {
				"motors": [
					"l_x","l_y","l_z","l_r1","l_r2","l_r3","l_r4","l_r5","l_r6",
					"r_x","r_y","r_z","r_r1","r_r2","r_r3","r_r4","r_r5","r_r6",
					"l_gripper","r_gripper"
				]
			}
		},
		"action_base": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "theta"]}
	}

	# === Remove previous dataset if exists ===
	dataset_path = Path(output_path)
	if dataset_path.exists():
		shutil.rmtree(dataset_path)

	# === Create dataset in video mode ===
	dataset = LeRobotDataset.create(
		repo_id="isaac2lerobot",
		root=dataset_path,
		fps=fps,
		use_videos=True,  # store image streams as mp4
		features=features,
		image_writer_processes=4,
		image_writer_threads=8,
	)
	with open(task_json, "r") as f1:
		task_data = json.load(f1)
	with open(goal_json, "r") as f2:
		goal_data = json.load(f2)
		goal_dict = {}
		for idx, sub_goal in enumerate(goal_data["goals"][0]):
			goal_dict[str(idx)] = np.array([sub_goal["action"], sub_goal["language"]])
		
	length = 0
	# === Open source HDF5 ===
	with h5py.File(hdf5_path, "r") as f:
		data = f["data"]

		# Go through all demos
		for idx, demo_key in enumerate(data.keys()):
			demo = data[demo_key]
			print(f"Processing {demo_key} ...")

			# Load Observations
			subtask_index = np.array(demo["observations"]["subtask_index"])
			mobile_base = np.array(demo["observations"]["mobile_base"])
			ee_pose = np.array(demo["observations"]["ee_6d_pos"])
			
			actions = np.array(demo["actions"]["ee_6D_pos"])
			actions_base = np.array(demo["actions"]["base"])

			# Images
			img_front = np.array(demo["observations"]["images"]["front"])
			img_wl = np.array(demo["observations"]["images"]["wrist_left"])
			img_wr = np.array(demo["observations"]["images"]["wrist_right"])

			# Ensure consistent length
			N = min(len(actions), len(actions_base), len(ee_pose))
			index = 0
			kitchen_type = {"island": 0,
							"l_shaped": 1,
							"peninsula": 2,
							"u_shaped": 3,
							"single_wall": 4
			}
			# Add all frames of this demo
			for i in range(N):
				index = i + length
				is_first = 0
				is_last = 0
				if i == 0:
					is_first = 1
				elif i == N-1:
					is_last = 1

				sample = {
					"kitchen_num": np.array([task_data["kitchen_num"]], dtype=np.int64),
					"kitchen_sub_num": np.array([int(task_data["kitchen_sub_num"])], dtype=np.int64),
					"kitchen_type": np.array([ kitchen_type[task_data["kitchen_type"]]],dtype=np.int64),
					"initial_pose": np.array(mobile_base[i], dtype=np.float32),
					"is_first": np.array([is_first]),
					"is_last": np.array([is_last]),
					"task": task_data["task_name"],
					"subtask_index": np.array([subtask_index[i]],dtype=np.int64),

#					"subtask_class": goal_dict[str(subtask_index[i])][0],
#					"subtask_language_instruction":goal_dict[str(subtask_index[i])][1],
					"observation.state.ee_pose": ee_pose[i],
					"observation.images.front": img_front[i],
					"observation.images.wrist_left": img_wl[i],
					"observation.images.wrist_right": img_wr[i],
					"action.original": np.array(actions[i], dtype=np.float32),
					"action_base": actions_base[i],
				}
				
				dataset.add_frame(sample)

			# Save after finishing one demo (one episode)
			dataset.save_episode()
			print(f"✅ Saved episode for {demo_key}")
			length += N

	print(f"\n🎉 Done! Merged dataset with videos saved to: {output_path}")


if __name__ == "__main__":
	hdf5_to_lerobot_with_videos(
		hdf5_path="/root/IsaacLab/Isaac-Kitchen-v1103/6.hdf5",
		task_json="/root/IsaacLab/scripts/simvla/goals/Isaac-Kitchen-v1103-00.json",
		goal_json="/root/IsaacLab/scripts/simvla/goals/Isaac-Kitchen-v1103-00.reloadable.json",
		output_path="/root/IsaacLab/datasets/lerobot/Isaac-Kitchen-v1103-00",
		fps=30,
	)
