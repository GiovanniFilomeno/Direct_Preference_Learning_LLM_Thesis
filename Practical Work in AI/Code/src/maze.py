import numpy as np

class Cell:
    def __init__(self):
        # Each cell has four walls: North, South, East, West (all closed by default)
        self.walls = {"N": True, "S": True, "E": True, "W": True}


class Maze:
    def __init__(self, nx, ny, ix, iy):
        self.nx = nx  # Number of columns
        self.ny = ny  # Number of rows
        self.ix = ix  # Start cell x-index
        self.iy = iy  # Start cell y-index
        # Initialize a ny x nx grid of Cells
        self.grid = [[Cell() for _il in range(nx)] for _ in range(ny)]
        self.make_maze()

    def is_wall(self, cx: int, cy: int) -> bool:
        """
        Return True if cell (cx, cy) is fully walled (all four directions).
        Useful for coarse path-finding that treats cells as blocked vs open.
        """
        cell = self.cell_at(cx, cy)
        return all(cell.walls.values())

    def make_maze(self):
        """
        Simple random maze initializer: for each cell, randomly choose
        between 'open' (no internal walls) and 'closed' (all walls set).
        Enforces border walls on the outer boundary and special handling
        for start/goal corners.
        """
        # 0 = open cell, 1 = closed cell (favor open cells 90% of the time)
        grid = np.random.choice([0, 1], size=(self.ny, self.nx), p=[0.9, 0.1])

        for y in range(self.ny):
            for x in range(self.nx):
                # Start from fully closed
                self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}
                if grid[y, x] == 0:
                    # Open the cell internally (remove internal walls)
                    self.grid[y][x].walls = {"N": False, "S": False, "E": False, "W": False}
                    # Keep border walls at the outer boundary
                    if y == 0:
                        self.grid[y][x].walls["N"] = True
                    elif x == 0:
                        self.grid[y][x].walls["W"] = True
                    elif y == self.ny - 1:
                        self.grid[y][x].walls["S"] = True
                    elif x == self.nx - 1:
                        self.grid[y][x].walls["E"] = True

        # Ensure the initial cell is open
        self.grid[self.iy][self.ix].walls = {"N": False, "S": False, "E": False, "W": False}

        # Corner constraints (keep perimeter closed where applicable)
        self.grid[0][0].walls["N"] = True  # top border at (0,0)
        self.grid[0][0].walls["W"] = True  # left border at (0,0)
        self.grid[self.ny - 1][self.nx - 1].walls["S"] = True  # bottom border at (nx-1, ny-1)
        self.grid[self.ny - 1][self.nx - 1].walls["E"] = True  # right border at (nx-1, ny-1)
        # Note: next two lines keep edges at opposite corners closed
        self.grid[0][self.ny - 1].walls["E"] = True  # (potential indexing mismatch with nx)
        self.grid[self.ny - 1][0].walls["S"] = True

    def make_maze_fail(self, step: int = 3):
        """
        Build a 'snake/serpentine' maze with outer perimeter closed.

        Parameters
        ----------
        step : int (default=3)
            How many consecutive rows the path spans before stepping down.
            - step = 1 → tight serpentine (every row)
            - step = 2 → descend every two rows
            - step = 3 → descend every three rows, etc.
        """
        print(f"Creating Path with step {step}")
        nx, ny = self.nx, self.ny

        # (1) Close all cells
        for y in range(ny):
            for x in range(nx):
                self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}

        # (2) Create serpentine path in blocks of `step` rows
        row = 0
        direction_right = True  # initial direction

        while row < ny:
            upper = row                        # first row of the block
            lower = min(row + step - 1, ny-1)  # last row of the block (inclusive)

            # (2a) Horizontal corridors within the block
            for r in range(upper, lower + 1):
                if direction_right:            # left → right
                    for x in range(nx - 1):
                        self.grid[r][x].walls["E"]   = False
                        self.grid[r][x+1].walls["W"] = False
                else:                          # right → left
                    for x in range(nx - 1, 0, -1):
                        self.grid[r][x].walls["W"]   = False
                        self.grid[r][x-1].walls["E"] = False

                # Vertical link to the next row *within* the same block
                if r < lower:
                    side = nx - 1 if direction_right else 0
                    self.grid[r][side].walls["S"]   = False
                    self.grid[r+1][side].walls["N"] = False

            # (2b) Link to the next block below
            if lower + 1 < ny:
                side = nx - 1 if direction_right else 0
                self.grid[lower][side].walls["S"]    = False
                self.grid[lower+1][side].walls["N"]  = False

            # (2c) Advance to next block and flip direction
            row += step
            direction_right = not direction_right

        # (3) Close the outer perimeter
        for x in range(nx):
            self.grid[0][x].walls["N"]    = True
            self.grid[ny-1][x].walls["S"] = True
        for y in range(ny):
            self.grid[y][0].walls["W"]    = True
            self.grid[y][nx-1].walls["E"] = True

        # (4) Open start (0,0) and goal (nx-1, ny-1) internally
        self.grid[0][0].walls.update({"E": False, "S": False})
        self.grid[ny-1][nx-1].walls.update({"N": False, "W": False})

    def cell_at(self, x, y):
        """Return the Cell at grid coordinates (x, y)."""
        return self.grid[y][x]

    def display(self):
        """
        Print an ASCII representation of the maze:
        '+' and '---' for horizontal walls, '|' for vertical walls.
        """
        maze_rows = []
        for y in range(self.ny):
            row_top = ''
            row_mid = ''
            for x in range(self.nx):
                cell = self.grid[y][x]
                # North wall
                row_top += '+---' if cell.walls['N'] else '+   '
                # West wall plus cell interior
                row_mid += '|   ' if cell.walls['W'] else '    '
            # East wall for the last cell in the row
            row_mid += '|' if self.grid[y][-1].walls['E'] else ' '
            maze_rows.append(row_top + '+')
            maze_rows.append(row_mid)
        # Bottom border
        row_bottom = ''
        for x in range(self.nx):
            row_bottom += '+---' if self.grid[-1][x].walls['S'] else '+   '
        row_bottom += '+'
        maze_rows.append(row_bottom)
        # Print
        for row in maze_rows:
            print(row)


# Example usage
if __name__ == "__main__":
    maze = Maze(10, 10, 0, 0)
    maze.make_maze()
