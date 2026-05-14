import numpy as np
from agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """
    Baseline safe agent.

    Stratégie :
    - aller vers le bord gauche pour éviter l'île centrale ;
    - remonter le long du bord gauche ;
    - une fois en haut, revenir vers le goal au centre.
    """

    def __init__(self):
        super().__init__()
        self.grid_size = (128, 128)
        self.goal_position = np.array([64, 127])
        self.np_random = np.random.default_rng()

    def act(self, observation: np.ndarray) -> int:
        x, y = observation[0], observation[1]

        
        if y >= 126:
            if x < 64:
                return 2  
            elif x > 64:
                return 6  
            else:
                return 8  

        
        if x > 3:
            return 6  

      
        return 0  

    def reset(self) -> None:
        pass

    def seed(self, seed=None) -> None:
        self.np_random = np.random.default_rng(seed)