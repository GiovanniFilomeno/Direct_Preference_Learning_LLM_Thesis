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

    def is_wall(self, cx: int, cy: int) -> bool:
        """
        Ritorna True se la cella (cx,cy) è completamente chiusa da muri
        in tutte e 4 le direzioni.  Utile per algoritmi di path‑finding
        grossolani che trattano le celle come passaggi/discese.
        """
        cell = self.cell_at(cx, cy)
        return all(cell.walls.values())    

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

    def make_maze_fail(self, step: int = 3):
        """
        Crea un labirinto a serpentina.
        
        Parametri
        ---------
        step : int, opzionale (default = 1)
            Ogni quante righe il percorso scende di livello.
            - step = 1  → serpentina fitta (originale)
            - step = 2  → il percorso attraversa due righe prima di scendere
            - step = 3  → tre righe, e così via
        """
        print(f"Creating Path with step {step}")
        nx, ny = self.nx, self.ny

        # 1) chiudi tutto
        for y in range(ny):
            for x in range(nx):
                self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}

        # 2) crea la serpentina con blocchi di `step` righe
        row = 0
        direction_right = True                # direzione iniziale

        while row < ny:
            upper = row                       # prima riga del blocco
            lower = min(row + step - 1, ny-1) # ultima riga del blocco

            # 2a) orizzontali dentro il blocco
            for r in range(upper, lower + 1):
                if direction_right:           # sinistra → destra
                    for x in range(nx-1):
                        self.grid[r][x].walls["E"]       = False
                        self.grid[r][x+1].walls["W"]     = False
                else:                         # destra → sinistra
                    for x in range(nx-1, 0, -1):
                        self.grid[r][x].walls["W"]       = False
                        self.grid[r][x-1].walls["E"]     = False

                # collegamento verticale alle righe successive *dello stesso* blocco
                if r < lower:
                    side = nx-1 if direction_right else 0
                    self.grid[r][side].walls["S"]        = False
                    self.grid[r+1][side].walls["N"]      = False

            # 2b) collegamento al blocco successivo
            if lower + 1 < ny:
                side = nx-1 if direction_right else 0
                self.grid[lower][side].walls["S"]        = False
                self.grid[lower+1][side].walls["N"]      = False

            # 2c) passa al blocco successivo, inverti direzione
            row += step
            direction_right = not direction_right

        # 3) chiudi il perimetro esterno
        for x in range(nx):
            self.grid[0][x].walls["N"]        = True
            self.grid[ny-1][x].walls["S"]     = True
        for y in range(ny):
            self.grid[y][0].walls["W"]        = True
            self.grid[y][nx-1].walls["E"]     = True

        # 4) sblocca start (0,0) e goal (nx-1, ny-1)
        self.grid[0][0].walls.update({"E": False, "S": False})
        self.grid[ny-1][nx-1].walls.update({"N": False, "W": False})


    # def make_maze_fail(self):
    #     """
    #     Crea un labirinto a serpentina con confini esterni chiusi.
    #     L'agente parte da (0,0) in alto a sinistra e arriva a (nx-1, ny-1) in basso a destra.
    #     """
    #     nx = self.nx
    #     ny = self.ny

    #     # 1) Chiudi tutte le celle (tutti i muri = True)
    #     for y in range(ny):
    #         for x in range(nx):
    #             self.grid[y][x].walls = {"N": True, "S": True, "E": True, "W": True}

    #     # 2) Crea un percorso "a serpentina" attraverso la griglia
    #     #    Riga pari (y%2==0): apri da x=0 a x=nx-1
    #     #    Riga dispari (y%2==1): apri da x=nx-1 a x=0
    #     #    E alla fine di ogni riga, apri un passaggio verso la riga successiva.
    #     for row in range(ny):
    #         if row % 2 == 0:
    #             # Riga pari => apri in orizzontale da sinistra (0) a destra (nx-1)
    #             for x in range(nx - 1):
    #                 self.grid[row][x].walls["E"] = False
    #                 self.grid[row][x + 1].walls["W"] = False
    #             # Se non siamo all'ultima riga, apri un passaggio giù (S)
    #             if row < ny - 1:
    #                 self.grid[row][nx - 1].walls["S"] = False
    #                 self.grid[row + 1][nx - 1].walls["N"] = False
    #         else:
    #             # Riga dispari => apri in orizzontale da destra (nx-1) a sinistra (0)
    #             for x in range(nx - 1, 0, -1):
    #                 self.grid[row][x].walls["W"] = False
    #                 self.grid[row][x - 1].walls["E"] = False
    #             # Se non siamo all'ultima riga, apri un passaggio giù (S)
    #             if row < ny - 1:
    #                 self.grid[row][0].walls["S"] = False
    #                 self.grid[row + 1][0].walls["N"] = False

    #     # 3) Ora chiudiamo "davvero" il perimetro esterno (il bounding box)
    #     #    in modo che l'agente non possa scappare fuori.
    #     for x in range(nx):
    #         self.grid[0][x].walls["N"]       = True  # riga in alto
    #         self.grid[ny - 1][x].walls["S"]  = True  # riga in basso
    #     for y in range(ny):
    #         self.grid[y][0].walls["W"]       = True  # colonna sinistra
    #         self.grid[y][nx - 1].walls["E"]  = True  # colonna destra

    #     # 4) Sblocca un po' i muri delle celle di start e goal
    #     #    ma lasciando comunque chiuso il perimetro in corrispondenza del bordo.
    #     #    Start: (0,0) in alto a sinistra
    #     self.grid[0][0].walls["N"] = True   # confine alto
    #     self.grid[0][0].walls["W"] = True   # confine sinistra
    #     self.grid[0][0].walls["E"] = False  # si può andare a destra
    #     self.grid[0][0].walls["S"] = False  # si può andare in basso

    #     #    Goal: (nx-1, ny-1) in basso a destra
    #     self.grid[ny - 1][nx - 1].walls["N"] = False
    #     self.grid[ny - 1][nx - 1].walls["W"] = False
    #     self.grid[ny - 1][nx - 1].walls["S"] = True   # confine in basso
    #     self.grid[ny - 1][nx - 1].walls["E"] = True   # confine a destra



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