import numpy as np

class Grid():
    def __init__(self, world_size=10, patches=10, max_food_quant=5, random_quant=False):
        self.world_size = world_size
        self.patches = patches
        self.max_food_quant = max_food_quant
        self.random_quant = random_quant
        self.setup_grid()

    def setup_grid(self):
        self.world = np.zeros((self.world_size, self.world_size))
        # a patch is marked in the world grid by the quantity of food stored
        for p in range(self.patches):
            x = np.random.randint(0, self.world_size)
            y = np.random.randint(0, self.world_size)
            if self.random_quant:
                q = np.random.randint(1, self.max_food_quant)
            else:
                q = self.max_food_quant
            self.world[x][y] = q

    def step(self, locations, random=False):
        # global reward
        rew = 0

        # explored locations
        obs = []

        for l in locations:
            x = l[0]
            y = l[1]

            if random:
                x = np.random.randint(0, self.world_size)
                y = np.random.randint(0, self.world_size)

            # collects food at location that agent picked if there is some
            q = self.world[x][y]
            if q > 0:
                rew += 1
                self.world[x][y] -= 1
            else:
                # TODO adjust punishment
                rew -= 0.1

            # store explored location
            obs.append([x, y, q])

        done = True
        if np.any(self.world):
            done = False

        return np.array(obs), rew, done