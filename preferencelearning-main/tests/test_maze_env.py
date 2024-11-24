import unittest
import numpy as np
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
        self.env = MazeEnv(sz=10, maze=maze, start=np.array([0.0, 0.0]), goal=np.array([0.75, 0.75]),
                 reward="distance", log=False, eval=False, dt=0.1, horizon=50, 
                 wall_penalty=10, slide=1, image_freq=20) 
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
        # Test recording a video of the agent interacting with the environment
        fig, ax = plt.subplots()
        ims = []

        self.env.reset()
        for i in range(1000):
            # Random Action
            action = np.random.uniform(-1, 1, size=(2,))
            # Fixed Action
#            action = np.array([1, 0.5])
            state, reward, done, truncated, infos = self.env.step(action)
#            print(state)
            maps = draw_map(1 / self.env.sz, self.env.maze)
            if self.env.Z is None:
                self.env.Z = self.env.reward_fn.compute_reward(self.env.points)  # negating reward to standardize plot
            maps.append(plt.contourf(self.env.X, self.env.Y, self.env.Z, 30, cmap='viridis_r'))

            # if self.env.Z is None:
            #     self.env.Z = self.env.reward_fn.compute_reward(self.env.points)  # negating reward to standardize plot            
#            plt.contourf(self.env.X, self.env.Y, self.env.Z, 30, cmap='viridis_r')
            # plot on top of the countour plot and return the image
            im = plt.scatter(state[0], state[1], c='r')
            # save the image to a folder
            im.figure.savefig(f'tests/video/state_{i}.png')

#            im = ax.scatter(state[0], state[1], c='r')
            ims.append([im])
            if done or truncated:
                break
        

        # Create an animation from stored images
        images = []
        for i in range(len(ims)):
            images.append(imageio.imread(f'tests/video/state_{i}.png'))
        imageio.mimsave('agent_interaction.gif', images, fps=30)
        # remove all the images
        for i in range(len(ims)):
            os.remove(f'tests/video/state_{i}.png')


    
    

if __name__ == '__main__':
    unittest.main()