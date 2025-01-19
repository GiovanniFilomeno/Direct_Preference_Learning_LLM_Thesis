import unittest
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from maze_env import MazeEnv, draw_map
from maze import Maze
import matplotlib.pyplot as plt
import imageio
import os

class TestMazeEnv(unittest.TestCase):

    def setUp(self):
        # Initialize the environment with necessary parameters
        maze = Maze(10, 10, 0, 0) # inizializzazione del labirinto 10x10 dimensione e 0-0 punto iniziale
        start_test = np.array([0.05, 0.05])
        # inizializza l'ambiente: sz=10 (dimensione griglia), start/goal sono le posizioni iniziali e finali 
        self.env = MazeEnv(sz=10, maze=maze, start=start_test, goal=np.array([0.45, 0.70]),
                 reward="distance", log=False, eval=False, dt=0.1, horizon=10, 
                 wall_penalty=10, slide=1, image_freq=100) 
        self.env.reset(state=start_test)

    def test_step(self):
        # Test the step function with a valid action
        action = np.array([0.5, 0.5]) # esegui una azione
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

            # Ora draw_map restituisce un Axes => lo chiamo "ax_map"
            ax_map = draw_map(1 / self.env.sz, self.env.maze, ax=ax)

            if self.env.Z is None:
                self.env.Z = self.env.reward_fn.compute_reward(self.env.points)  

            # Uso "ax_map.contourf" (o "ax.contourf")
            cf = ax_map.contourf(self.env.X, self.env.Y, self.env.Z, 30, cmap='viridis_r')

            # Scatter plot per lo stato (uso sempre lo stesso ax)
            im = ax.scatter(state[0], state[1], c='r')

            # Salvo immagine
            im.figure.savefig(os.path.join(video_dir, f'state_{i}.png'))
            ims.append([im, cf])  # puoi anche solo [im] se vuoi

            if done or truncated:
                break

            # Pulisco l'axes per il prossimo frame (se vuoi sovrascrivere)
            ax.cla()

        # Creare l'animazione dalle immagini salvate
        images = []
        for i in range(len(ims)):
            images.append(imageio.v2.imread(os.path.join(video_dir, f'state_{i}.png')))
        imageio.mimsave('agent_interaction_fail.gif', images, duration=33)

        # Rimuovere le immagini salvate
        for i in range(len(ims)):
            os.remove(os.path.join(video_dir, f'state_{i}.png'))

        # Rimuovi la directory `video` se vuota
        os.rmdir(video_dir)

if __name__ == '__main__':
    unittest.main()
