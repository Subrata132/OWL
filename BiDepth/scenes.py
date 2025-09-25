import numpy as np
import random
import json
from typing import List


DIRECTION_MAPPING = [
    {1: "right", -1: "left"},
    {1: "front", -1: "behind"},
    {1: "above", -1: "below"}
]


class GibsonHouse:
    _habitat_order = [0, 2, 1]
    def __init__(self, gibson_path: str) -> None:
        with open(gibson_path) as f:
            self.gibson = json.load(f)

        self.house_idx = 0
    
    def set_house_idx(self, house_idx):
        self.house_idx = house_idx
    
    def get_house_id(self, house_idx=None):
        return self.gibson[self.house_idx]["id"]

    def get_room_num(self):
        return self.gibson[self.house_idx]['stats']['room']

    def get_room_info(self, index):
        assert index is None
        return self.gibson[self.house_idx]['stats']

    def compute_direction_distance(self, point_A, point_B):
        difference = point_B - point_A
        distance = np.linalg.norm(difference)

        direction = np.sign(difference)
        direction = [DIRECTION_MAPPING[i][val] for i, val in enumerate(direction)]
        return distance, direction

    def generate_coordinate(self, sim, prev_agent_position=None):
        
        if prev_agent_position is None:
            agent_coordinate = sim.pathfinder.get_random_navigable_point()[self._habitat_order]
        else:
            agent_coordinate = prev_agent_position[self._habitat_order]
        sensor_coordinate = agent_coordinate + np.array([0.0, 0.0, 1.5])
        
        soundsource_coordinate = sim.pathfinder.get_random_navigable_point()[self._habitat_order]
        soundsource_coordinate[2] += random.uniform(0, 3)
        if np.any(soundsource_coordinate == sensor_coordinate): return None
        
        distance, direction = self.compute_direction_distance(sensor_coordinate, soundsource_coordinate)
        if distance > 10: return None  
        # (X, Y, Z) --> (X, Z, Y)
        return {
            "soundsource_coordinate": soundsource_coordinate[self._habitat_order],
            "agent_coordinate": agent_coordinate[self._habitat_order],
            "sensor_coordinate": sensor_coordinate[self._habitat_order],
            "direction": direction,
            "distance": distance,
            "sensor_room_id": None,
            "soundsource_room_id": None,
        }


class MP3DHouse:
    _habitat_order = [0, 2, 1]
    def __init__(self, mp3d_path: str) -> None:
        with open(mp3d_path) as f:
            self.mp3d = json.load(f)

        self.house_idx = 0
        self.room_idx = 0
    
    def set_house_idx(self, house_idx):
        self.house_idx = house_idx
        self.room_corner = {}
        for room in self.mp3d[self.house_idx]['rooms']: # (X, Y, Z)
            left_bottom_corner = np.array(list(room['left_bottom_vertex'].values()))
            right_top_corner = np.array(list(room['right_top_vertex'].values()))
            self.room_corner[room['region_index']] = (room['label'], left_bottom_corner, right_top_corner, room['height'])

    def get_house_id(self):
        return self.mp3d[self.house_idx]['house_id']

    def get_room_num(self):
        return len(self.mp3d[self.house_idx]['rooms'])

    def get_room_info(self, index: int = None):
        if index == -1:
            return ''
        if index is not None:
            info = self.mp3d[self.house_idx]['rooms'][index]
        else:
            info = self.mp3d[self.house_idx]['rooms'][self.room_idx]
        return f'region_index: {info["region_index"]}; room_type: {info["label"]}; height: {info["height"]}.'
    
    def compute_direction_distance(self, point_A, point_B):
        difference = point_B - point_A
        distance = np.linalg.norm(difference)

        direction = np.sign(difference)
        direction = [DIRECTION_MAPPING[i][val] for i, val in enumerate(direction)]
        return distance, direction

    def fetch_room_type_and_height(self, coordinate):
        for key, value in self.room_corner.items():
            room_type, left_bottom_corner, right_top_corner, height = value
            if left_bottom_corner[0] <= coordinate[0] <= right_top_corner[0] and \
                left_bottom_corner[1] <= coordinate[1] <= right_top_corner[1]:
                return key, height
        
        return -1, None  # agent in corridor

    def get_random_coordinate_in_room(self, room_id):
        left_bottom_corner = self.room_corner[room_id][1]
        right_top_corner = self.room_corner[room_id][2]
        return np.array([
            random.uniform(left_bottom_corner[0], right_top_corner[0]),
            random.uniform(left_bottom_corner[1], right_top_corner[1]),
            left_bottom_corner[2] + 0.05
        ]) # (X, Y, Z)

    def generate_coordinate(self, sim, is_same=True, allow_corridor=False, prev_agent_position=None):
        while True:
            selected_room_id = random.choice(list(self.room_corner.keys()))
            if prev_agent_position is None:
                if allow_corridor:
                    agent_coordinate = sim.pathfinder.get_random_navigable_point()[self._habitat_order]
                    room_id, height = self.fetch_room_type_and_height(agent_coordinate)
                else:
                    agent_coordinate = self.get_random_coordinate_in_room(selected_room_id)
                    room_id, height = selected_room_id, self.room_corner[selected_room_id][3]
                    if height < 1.5:
                        continue
            else:
                agent_coordinate = prev_agent_position[self._habitat_order]
                room_id, height = self.fetch_room_type_and_height(agent_coordinate)
            sensor_coordinate = agent_coordinate + np.array([0.0, 0.0, 1.5])
            break   

        while True:
            if room_id == -1:
                if is_same:
                    soundsource_coordinate = sim.pathfinder.get_random_navigable_point()[self._habitat_order]
                    room_id2, height2 = self.fetch_room_type_and_height(soundsource_coordinate)
                else:
                    room_id2 = random.choice(list(self.room_corner.keys()))
                    soundsource_coordinate = self.get_random_coordinate_in_room(room_id2)
                    height2 = self.room_corner[room_id2][3]
            else:
                if is_same:
                    soundsource_coordinate = self.get_random_coordinate_in_room(room_id)   
                    room_id2, height2 = room_id, height
                else:
                    room_id2 = random.choice(list(self.room_corner.keys()) + [-1]) 
                    if room_id2 == -1:
                        soundsource_coordinate = sim.pathfinder.get_random_navigable_point()[self._habitat_order]
                    else:
                        soundsource_coordinate = self.get_random_coordinate_in_room(room_id2)
                height2 = self.room_corner[room_id2][3] if room_id2 != -1 else 3

            soundsource_coordinate[2] += random.uniform(0, height2) if height2 is not None else random.uniform(0, 3)
            if np.any(soundsource_coordinate == sensor_coordinate): continue

            distance, direction = self.compute_direction_distance(sensor_coordinate, soundsource_coordinate)
            if distance > 10: return None            
            
            break        

        return {
            "soundsource_coordinate": soundsource_coordinate[self._habitat_order],
            "agent_coordinate": agent_coordinate[self._habitat_order],
            "sensor_coordinate": sensor_coordinate[self._habitat_order],
            "direction": direction,
            "distance": distance,
            "sensor_room_id": room_id,
            "soundsource_room_id": room_id2,
        }