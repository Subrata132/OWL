import os
import math
from datetime import datetime
import logging
import numpy as np
import magnum as mn
import quaternion
import habitat_sim.sim
from habitat_sim.utils.common import quat_from_angle_axis

logger = logging.getLogger(f"GENERATOR")

TYPE = [
    habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono,
    habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural,
    habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
]

CHANNELS = [1, 2, 9]

def make_configuration(dataset, scene):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    
    if dataset == 'mp3d':
        backend_cfg.scene_id = f"data/scene_datasets/mp3d/{scene}/{scene}.glb"
        backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    elif dataset == 'gibson':
        backend_cfg.scene_id = f"data/scene_datasets/gibson/{scene}/{scene}.glb"
        backend_cfg.scene_dataset_config_file = "data/scene_datasets/gibson/gibson_semantic.scene_dataset_config.json"

    backend_cfg.enable_physics = False
    backend_cfg.load_semantic_mesh = True

    # agent configuration
    agent_config = habitat_sim.AgentConfiguration()
    
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = 'depth_camera'
    depth_sensor_cfg.resolution = (512, 256)
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.hfov = mn.Deg(20)
    depth_sensor_cfg.position = np.array([0.0, 1.5, 0.0])
    agent_config.sensor_specifications = [depth_sensor_cfg]
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_config])

    return cfg


def configure_audio_sensor_spec(channel_type, position=[0.0, 1.5, 0.0]):
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.enableMaterials = True
    audio_sensor_spec.channelLayout.type = TYPE[channel_type]
    audio_sensor_spec.channelLayout.channelCount = CHANNELS[channel_type]

    audio_sensor_spec.position = position
    audio_sensor_spec.acousticsConfig.sampleRate = 32000

    audio_sensor_spec.acousticsConfig.indirect = True

    return audio_sensor_spec


def configure_agent(sim, agent_coordinate):
    agent = sim.get_agent(0)
    new_state = agent.get_state()

    new_state.position = agent_coordinate
    new_state.sensor_states = {}
    agent.set_state(new_state, True)
    return agent


def get_res_angles_for(fov):
    if fov == 20:
        # resolution = (384, 64)
        resolution = (384, 128)
        angles = [170, 150, 130, 110, 90, 70, 50, 30, 10, 350, 330, 310, 290, 270, 250, 230, 210, 190]
    elif fov == 30:
        resolution = (384, 128)
        angles = [0, 330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30]
    elif fov == 60:
        # resolution = (256, 128)
        resolution = (384, 128)
        angles = [0, 300, 240, 180, 120, 60]
    elif fov == 90:
        # resolution = (256, 256)
        resolution = (384, 128)
        angles = [0, 270, 180, 90]
    else:
        raise ValueError

    return resolution, angles


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def visual_render(sim, receiver):
    depth_panorama = []
    angles = get_res_angles_for(20)[1]
    for angle in angles:
        agent = sim.get_agent(0)
        new_state = sim.get_agent(0).get_state()
        new_state.position = receiver
        new_state.rotation = quat_from_angle_axis(math.radians(angle), np.array([0, 1, 0]))
        new_state.sensor_states = {}
        agent.set_state(new_state, True)
        observation = sim.get_sensor_observations()
        depth_panorama.append(normalize_depth(observation['depth_camera']))
    depth_panorama = np.concatenate(depth_panorama, axis=1)
    return depth_panorama