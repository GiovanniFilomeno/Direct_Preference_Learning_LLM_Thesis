import unittest
import numpy as np
import os
import sys

module_path = os.path.abspath("../src")
if module_path not in sys.path:
    sys.path.append(module_path)

from maze_env import MazeEnv, draw_map
from maze import Maze
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import os

class TestMazeEnv(unittest.TestCase):

    def setUp(self):
        # Initialize the environment with necessary parameters
        maze = Maze(10, 10, 0, 0) # inizilizzazione del labirinto 10x10 dimensione e 0-0 punto iniziale

        # inizializza l'ambiente: sz=10 (dimensione griglia), start/goal sono le posizioni iniziali e finali 
        self.env = MazeEnv(sz=10, maze=maze, start=np.array([0.15, 0.15]), goal=np.array([0.45, 0.70]),
                 reward="distance", log=False, eval=False, dt=0.1, horizon=70, 
                 wall_penalty=10, slide=1, image_freq=100) 
        self.env.reset()

    def test_step(self):
        # Test the step function with a valid action
        action = np.array([0.5, 0.5]) # esegui una azioni
        state, reward, done, truncated, infos = self.env.step(action)
        
        # Check if the state is updated correctly
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2,))
        
        # Check if reward is a float
        self.assertIsInstance(reward, float)
        
        # Check if done is a boolean
        self.assertIsInstance(done, bool)
        
        # Check if infos is a dictionary
        self.assertIsInstance(infos, dict)
        
        # Check if the environment correctly identifies the end of an episode
        self.env.counter = self.env.horizon
        _, _, done, truncated, _ = self.env.step(action)
        self.assertTrue(done)

    def test_reset(self):
        # Test the reset function
        state = self.env.reset()
        
        # Check if the state is reset correctly
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2,))
        
        # Check if the counter is reset
        self.assertEqual(self.env.counter, 0)

    def test_collision_detection(self):
        # Test collision detection logic
        action = np.array([0.5, 0.5])
        state, reward, done, truncated, infos = self.env.step(action)
        
        # Check if the state is updated correctly
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (2,))
        
        # Check if reward is a float
        self.assertIsInstance(reward, float)
        
        # Check if done is a boolean
        self.assertIsInstance(done, bool)
        
        # Check if infos is a dictionary
        self.assertIsInstance(infos, dict)

    def test_record_video(self):
        # Assicurati che la directory video esista
        video_dir = os.path.join(os.path.dirname(__file__), 'video')
        os.makedirs(video_dir, exist_ok=True)

        print(f"Directory created or already existing: {video_dir}")


        fig, ax = plt.subplots()
        ims = []

        self.env.reset()
        for i in range(1000):
            # Azione casuale
            action = np.random.uniform(-1, 1, size=(2,))
            state, reward, done, truncated, infos = self.env.step(action)

            # Disegna mappa
            maps = draw_map(1 / self.env.sz, self.env.maze)
            if self.env.Z is None:
                self.env.Z = self.env.reward_fn.compute_reward(self.env.points)  # Negate per standardizzare

            maps.append(plt.contourf(self.env.X, self.env.Y, self.env.Z, 30, cmap='viridis_r'))

            # Scatter plot per lo stato
            im = plt.scatter(state[0], state[1], c='r')
            im.figure.savefig(os.path.join(video_dir, f'state_{i}.png'))
            ims.append([im])
            if done or truncated:
                break

        # Creare l'animazione dalle immagini salvate
        images = []
        for i in range(len(ims)):
            images.append(imageio.v2.imread(os.path.join(video_dir, f'state_{i}.png')))
        imageio.mimsave('agent_interaction_fail.gif', images, duration=33)  # Usa `duration` invece di `fps`

        # Rimuovere le immagini salvate
        for i in range(len(ims)):
            os.remove(os.path.join(video_dir, f'state_{i}.png'))

        # Rimuovi la directory `video` se vuota
        os.rmdir(video_dir)



    
    

if __name__ == '__main__':
    unittest.main()