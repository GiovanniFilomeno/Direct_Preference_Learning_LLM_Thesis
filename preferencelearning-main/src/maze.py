import numpy as np

class Cell:
    def __init__(self):
        self.walls = {"N": True, "S": True, "E": True, "W": True}  # Initialize walls


class Maze:
    def __init__(self, nx, ny, ix, iy):
        self.nx = nx  # Number of columns
        self.ny = ny  # Number of rows
        self.ix = ix  # Initial x-coordinate
        self.iy = iy  # Initial y-coordinate
        self.grid = [[Cell() for _ in range(nx)] for _ in range(ny)]  # Initialize the maze grid
        self.make_maze()

    def make_maze(self):
        # Simple maze generation algorithm (e.g., recursive division, random walk, etc.)
        # Here, we'll use a placeholder for a simple maze generation
#        grid = np.random.randint(2, size=(self.ny, self.nx))  # Randomly generate walls and paths
        grid = np.random.choice([0, 1], size=(self.ny, self.nx), p=[0.9, 0.1])  # Randomly generate walls and paths

        for y in range(self.ny):
            for x in range(self.nx):
                self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}
                if grid[y, x] == 0:
                    self.grid[y][x].walls = {"N": False, "S": False, "E": False, "W": False}
                    # check if the cell is on the North border
                    if y == 0:
                        self.grid[y][x].walls["N"] = True
                    elif x == 0:
                        self.grid[y][x].walls["W"] = True
                    elif y == self.ny - 1:
                        self.grid[y][x].walls["S"] = True
                    elif x == self.nx - 1:
                        self.grid[y][x].walls["E"] = True

        self.grid[self.iy][self.ix].walls = {"N": False, "S": False, "E": False, "W": False}
        self.grid[0][0].walls["N"] = True # bottom left corner cell's south wall
        self.grid[0][0].walls["W"] = True # bottom left corner cell's west wall
        self.grid[self.ny - 1][self.nx - 1].walls["S"] = True # top right corner cell's north wall
        self.grid[self.ny - 1][self.nx - 1].walls["E"] = True # top right corner cell's east wall
        self.grid[0][self.ny - 1].walls["E"] = True # bottom right corner cell's east wall
        self.grid[self.ny - 1][0].walls["S"] = True


    def cell_at(self, x, y):
        return self.grid[y][x]

    def display(self):
        raise NotImplementedError("Display method not implemented yet")

# Example usage
if __name__ == "__main__":
    maze = Maze(10, 10, 0, 0)
    maze.make_maze()