# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:55:02 2017

@author: nehatambe
"""
import numpy as np
from data_utils import load_image_data
import time
import random
import yaml
from memory import Experience, Memory

ACTIONS = ['left', 'right','center']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 1000000
next_image_counter=0

class TrainingInfo:

    INFO_FILE = 'training-info.yaml'

    def __init__(self):
        self.data = {
            'episode': 1,
            'frames': 0,
            'mean_training_time': 1.0,
            'batches_per_frame': 1
        }

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self):
        with open(self.INFO_FILE, 'w') as file:
            yaml.safe_dump(self.data, file, default_flow_style=False)
            
class State:

    def __init__(self, data: np.array):
        # Convert to 0 - 1 ranges
        self.data = data.astype(np.float32) / 255
        # Z-normalize
        self.data = (self.data - np.mean(self.data)) / np.std(self.data, ddof=1)
        # Add channel dimension
        if len(self.data.shape) < 3:
            self.data = np.expand_dims(self.data, axis=2)

class StateAssembler:

    FRAME_COUNT = 4

    def __init__(self):
        self.cache = deque(maxlen=self.FRAME_COUNT)

    def assemble_next(self, unity_image: np.array) -> State:
        self.cache.append(unity_image)
        # If cache is still empty, put this image in there multiple times
        while len(self.cache) < self.FRAME_COUNT:
            self.cache.append(unity_image)
        images = np.stack(self.cache, axis=2)
        return State(images)
    
class RLLearner:
    MODEL_PATH = 'C:/virtual_car/CPSC587DATA/RecordedImg'
    def __init__(self, model):
        self.data, self.labels = load_image_data()
        self.training_info = TrainingInfo(True)
        self.batch_size = 100
        if batches_per_frame:
            self.training_info['batches_per_frame'] = batches_per_frame        
        self.model = model
        
    def _calc_reward(label) -> float:
        if label=='OFF_LEFT':
            return 0
        elif label=='OFF_RIGHT':
            return 0
        else:    
            return 1

    def read_image(self, t:int) -> (State, int):
        unity_image = self.data[t]
        image_label = self.labels[t]
        reward = self._calc_reward(image_label)
        state = self.assembler.assemble_next(unity_image)
        return state, reward

    def start_training(self):
        start_episode = self.training_info['episode']
        frames_passed = self.training_info['frames']
        train_frames = 1000000 
        t = 0;
        for episode in range(start_episode, episodes + 1):
            # Set initial state
            state = self.read_image(t)
            episode_start_time = time.time()
            while t < train_frames:
                t += 1
                random_probability = self.random_action_policy.get_probability(frames_passed)
                if random.random() < random_probability:
                    action = self.random_action_policy.sample_action(self.action_type)
                else:
                    # noinspection PyTypeChecker
                    action = self.action_type.from_code(np.argmax(self._predict(state)))
                
                for _ in range(self.training_info['batches_per_frame']):
                    self._train_minibatch()
                new_state, reward = self.read_image(t)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)

                state = new_state
                frames_passed += 1

                # Print status
                time_since_failure = time.time() - episode_start_time
                print('Episode {}, Total frames {}, ε={:.4f}, Reward {:.4f}, '
                      '{:.0f}s since failure'
                      .format(episode, frames_passed, random_probability,reward,time_since_failure), end='\r')

                # Save model after a fixed amount of frames
                if frames_passed % 1000 == 0:
                    self.training_info['episode'] = episode
                    self.training_info['frames'] = frames_passed
                    self.training_info['mean_training_time'] = self.mean_training_time.get()
                    self.training_info.save()
                    self.model.save(self.MODEL_PATH)
                    
    
    def _predict(self, state: State) -> np.array:
        # Add batch dimension
        x = np.expand_dims(state.data, axis=0)
        return self.model.predict_on_batch(x)[0]

    def _predict_multiple(self, states: Iterable[State]) -> np.ndarray:
        x = np.stack(state.data for state in states)
        return self.model.predict_on_batch(x)

    def _generate_minibatch(self) -> (np.ndarray, np.ndarray):
        batch = self.memory.random_sample(self.batch_size)

        # Estimate Q values using current model
        from_state_estimates = self._predict_multiple(experience.from_state for experience in batch)
        to_state_estimates = self._predict_multiple(experience.to_state for experience in batch)

        # Create arrays to hold input and expected output
        x = np.stack(experience.from_state.data for experience in batch)
        y = from_state_estimates

        # Reestimate y values where new reward is known
        for index, experience in enumerate(batch):
            new_y = experience.reward
            if not experience.to_state.is_terminal:
                new_y += self.discount * np.max(to_state_estimates[index])
            y[index, experience.action.get_code()] = new_y

        return x, y

    def _train_minibatch(self):
        if len(self.memory) < 1:
            return
        start = time.perf_counter()
        x, y = self._generate_minibatch()
        self.model.train_on_batch(x, y)
        end = time.perf_counter()
        self.mean_training_time.add(end - start)

    def predict(self):
        signal.signal(signal.SIGINT, self.stop)
        while True:
            state = self.environment.read_sensors(self.image_size, self.image_size)[0]
            while not state.is_terminal:
                action = self.action_type.from_code(np.argmax(self._predict(state)))
                self.environment.write_action(action)
                # Wait as long as we usually need to wait due to training
                time.sleep(self.training_info['batches_per_frame'] *
                           self.training_info['mean_training_time'])
                new_state, reward = self.environment.read_sensors(self.image_size, self.image_size)
                experience = Experience(state, action, reward, new_state)
                self.memory.append_experience(experience)
                state = new_state

                if self.should_exit:
                    sys.exit(0)
        
        
    def __init__(self, host: str, port: int):
        self.assembler = StateAssembler()
        
        
        