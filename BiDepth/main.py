import os
import json
import random
import string
import argparse
import logging

import numpy as np
import soundfile as sf

import quaternion
import habitat_sim.sim

from scenes import MP3DHouse, GibsonHouse
from generator import *


STRINGS = string.digits + string.ascii_letters
TYPE = ["BINAURAL"]


def build_logging(output_dir, house_id, ss_room_id=None, sensor_room_id=None, channel_type=1):
    uniq_id = ''.join(random.choices(STRINGS, k=5))

    log_folder = os.path.join(output_dir, "logs", TYPE[channel_type], house_id)
    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(log_folder, f"SS{ss_room_id}-RC{sensor_room_id}-{uniq_id}.log")

    logger = logging.getLogger(f"{TYPE[channel_type]}-{uniq_id}") 
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())  

    fh = logging.FileHandler(log_filename, 'w')  
    fh.setLevel(os.environ.get("LOGLEVEL", "INFO").upper()) 

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger, fh, uniq_id


def main(args):
    os.chdir(args.repo_root)

    house_nums = {'mp3d': 90, 'gibson' : {'train': 75, 'val': 15, 'test': 15 }}    

    if args.dataset == 'mp3d':
        # initialization: load mp3d and audioset information
        scene_datasets = MP3DHouse(args.mp3d_path)
        house_num = house_nums['mp3d']
    elif args.dataset == 'gibson':
        scene_datasets = GibsonHouse(os.path.join(args.gibson_path, f'{args.gibson_split}.json'))
        house_num = house_nums['gibson'][args.gibson_split]
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    # 90 house in total
    for house_id in range(house_num):
        scene_datasets.set_house_idx(house_id)
        scene = scene_datasets.get_house_id()
        cfg = make_configuration(args.dataset, scene)
        
        sim = habitat_sim.Simulator(cfg)

        if args.dataset == 'mp3d':
            # set navmesh path for searching for navigable points
            sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/mp3d/{scene}/{scene}.navmesh"))
        elif args.dataset == 'gibson':
            sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/gibson/{scene}/{scene}.navmesh"))
            
        audio_sensor_spec = configure_audio_sensor_spec(args.channel_type)
        sim.add_sensor(audio_sensor_spec)
        
        main(args, cfg, sim, audio_sensor_spec, scene_datasets)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str,
                        default='/path/to/sound-spaces',
                        help='path to sound-spaces repo')

    parser.add_argument('--dataset', type=str, default='mp3d', help="mp3d or gibson")
    parser.add_argument('--gibson_split', type=str, default='train', help="train, val, or test")
    parser.add_argument('--gibson_path', type=str, help='path/to/gibson/',
                            default='/home/zhisheng/data/sound-spaces/scene_datasets/gibson/splits_medium')
    parser.add_argument('--mp3d_path', type=str,
                        default='/path/to/MatterPort3D_room_descriptor.json',
                        help="/path/to/mp3d_house_info")

    parser.add_argument('--channel_type', type=int, default=1, help="0: mono, 1: binaural, 2: ambisonics") # mono & ambisonics not implemented
    parser.add_argument('--simulation_per_room', type=int, default=10)
    parser.add_argument('--prob_same_room', type=float, default=0.75, help="Probability of agent and ss being in the same house")
    parser.add_argument('--corridor_prob', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42,)

    parser.add_argument("--output_dir", type=str, default='/path/to/RIR/ourput/folder')

    return parser.parse_args()


if __name__ == '__main__':
    parser = get_parser()
    main(parser)