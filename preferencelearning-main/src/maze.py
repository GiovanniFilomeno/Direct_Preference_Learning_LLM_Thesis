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
        self.grid = [[Cell() for _il  in range(nx)] for _ in range(ny)]  # Initialize the maze grid
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

    def make_maze_fail(self):
        # Initialize the grid with walls
        grid = np.ones((self.ny, self.nx))  # 1 represents walls, 0 represents paths

        # Create a complex path
        # Open a winding path through the grid
        grid[1:9, 1] = 0  # Vertical passage
        grid[8, 1:5] = 0  # Horizontal passage
        grid[4:8, 4] = 0  # Another vertical passage
        grid[4, 2:4] = 0  # Horizontal connection

        # Add a concave obstacle
        grid[5, 3:6] = 1  # Base of the obstacle
        grid[6, 5] = 1    # Concave part

        # Add additional obstacles to make the maze complex
        grid[2:4, 3] = 1  # Vertical block
        grid[6:8, 2] = 1  # Vertical block near the passage

        # Initialize walls for all cells
        for y in range(self.ny):
            for x in range(self.nx):
                self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}

        # Update walls based on grid
        for y in range(self.ny):
            for x in range(self.nx):
                if grid[y, x] == 0:
                    # Open walls to adjacent path cells
                    if y > 0 and grid[y - 1, x] == 0:
                        self.grid[y][x].walls["N"] = False
                        self.grid[y - 1][x].walls["S"] = False
                    if y < self.ny - 1 and grid[y + 1, x] == 0:
                        self.grid[y][x].walls["S"] = False
                        self.grid[y + 1][x].walls["N"] = False
                    if x > 0 and grid[y, x - 1] == 0:
                        self.grid[y][x].walls["W"] = False
                        self.grid[y][x - 1].walls["E"] = False
                    if x < self.nx - 1 and grid[y, x + 1] == 0:
                        self.grid[y][x].walls["E"] = False
                        self.grid[y][x + 1].walls["W"] = False

        # Optionally, set the starting cell to have no walls
        self.grid[self.iy][self.ix].walls = {"N": False, "S": False, "E": False, "W": False}

    


    def cell_at(self, x, y):
        return self.grid[y][x]

    def display(self):
        maze_rows = []
        for y in range(self.ny):
            row_top = ''
            row_mid = ''
            for x in range(self.nx):
                cell = self.grid[y][x]
                # Add the north wall
                if cell.walls['N']:
                    row_top += '+---'
                else:
                    row_top += '+   '
                # Add the west wall and space for the cell
                if cell.walls['W']:
                    row_mid += '|   '
                else:
                    row_mid += '    '
            # Close the east wall of the last cell
            if self.grid[y][-1].walls['E']:
                row_mid += '|'
            else:
                row_mid += ' '
            maze_rows.append(row_top + '+')
            maze_rows.append(row_mid)
        # Add the bottom border
        row_bottom = ''
        for x in range(self.nx):
            if self.grid[-1][x].walls['S']:
                row_bottom += '+---'
            else:
                row_bottom += '+   '
        row_bottom += '+'
        maze_rows.append(row_bottom)
        # Print the maze
        for row in maze_rows:
            print(row)


# Example usage
if __name__ == "__main__":
    maze = Maze(10, 10, 0, 0)
    maze.make_maze()