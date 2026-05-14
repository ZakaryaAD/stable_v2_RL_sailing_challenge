import numpy as np
from evaluator.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """
    Agent v2 : waypoint + wind-aware one-step planning.

    Objectif :
    - contourner l'île par la gauche, mais moins largement que v1 ;
    - utiliser le vent et la vitesse pour choisir l'action qui donne le meilleur progrès ;
    - éviter les cellules de l'île grâce à la carte fournie dans l'observation.

    Ce n'est pas encore notre agent RL final.
    C'est une baseline heuristique forte, qui servira aussi de fallback pour le futur Q-learning.
    """

    def __init__(self):
        super().__init__()
        self.grid_size = (128, 128)
        self.goal_position = np.array([64.0, 127.0])
        self.np_random = np.random.default_rng()

        self.directions = np.array([
            [0, 1],      # 0: North
            [1, 1],      # 1: Northeast
            [1, 0],      # 2: East
            [1, -1],     # 3: Southeast
            [0, -1],     # 4: South
            [-1, -1],    # 5: Southwest
            [-1, 0],     # 6: West
            [-1, 1],     # 7: Northwest
            [0, 0],      # 8: Stay
        ], dtype=float)

        self.boat_performance = 0.4
        self.max_speed = 8.0
        self.inertia_factor = 0.3

    def act(self, observation: np.ndarray) -> int:
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        position = np.array([x, y], dtype=float)
        velocity = np.array([vx, vy], dtype=float)
        wind = np.array([wx, wy], dtype=float)

        world_map = self._extract_world_map(observation)

        target = self._choose_target(position)

        best_action = 0
        best_score = -1e18

        for action in range(8):  # on évite stay sauf cas spécial
            direction = self.directions[action]
            new_velocity = self._predict_velocity(velocity, wind, direction)
            new_position = position + new_velocity
            new_position = np.clip(new_position, [0, 0], [127, 127])

            score = self._score_action(
                position=position,
                new_position=new_position,
                velocity=new_velocity,
                target=target,
                world_map=world_map,
                action=action,
            )

            if score > best_score:
                best_score = score
                best_action = action

        return int(best_action)

    def _choose_target(self, position: np.ndarray) -> np.ndarray:
        x, y = position

        # Phase 1 : se décaler à gauche avant la pointe de l'île.
        if y < 18:
            return np.array([31.0, 18.0])

        # Phase 2 : longer l'île par la gauche.
        # L'île commence autour de x=38, donc x=31/32 laisse une marge.
        if y < 90:
            return np.array([31.0, 92.0])

        # Phase 3 : revenir progressivement vers le goal.
        return self.goal_position.copy()

    def _score_action(
        self,
        position: np.ndarray,
        new_position: np.ndarray,
        velocity: np.ndarray,
        target: np.ndarray,
        world_map: np.ndarray,
        action: int,
    ) -> float:
        old_dist_target = np.linalg.norm(target - position)
        new_dist_target = np.linalg.norm(target - new_position)

        old_dist_goal = np.linalg.norm(self.goal_position - position)
        new_dist_goal = np.linalg.norm(self.goal_position - new_position)

        progress_target = old_dist_target - new_dist_target
        progress_goal = old_dist_goal - new_dist_goal

        score = 10.0 * progress_target + 2.0 * progress_goal

        # Bonus léger pour avancer vers le haut.
        score += 0.5 * velocity[1]

        # Pénalité si on part vers le bas.
        if velocity[1] < 0:
            score -= 5.0

        # Sécurité île / bord.
        if self._is_unsafe(new_position, world_map):
            score -= 10_000.0

        # Éviter de trop coller à la zone de l'île.
        x, y = new_position
        if 35 <= x <= 93 and 15 <= y <= 88:
            score -= 300.0

        # Éviter stay indirectement.
        if action == 8:
            score -= 100.0

        return score

    def _predict_velocity(
        self,
        current_velocity: np.ndarray,
        wind: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        wind_norm = np.linalg.norm(wind)

        if wind_norm <= 1e-12:
            new_velocity = self.inertia_factor * current_velocity
        else:
            wind_normalized = wind / wind_norm

            direction_norm = np.linalg.norm(direction)
            if direction_norm <= 1e-12:
                direction_normalized = np.array([1.0, 0.0])
            else:
                direction_normalized = direction / direction_norm

            efficiency = self._sailing_efficiency(direction_normalized, wind_normalized)

            theoretical_velocity = (
                direction * efficiency * wind_norm * self.boat_performance
            )

            speed = np.linalg.norm(theoretical_velocity)
            if speed > self.max_speed:
                theoretical_velocity = theoretical_velocity / speed * self.max_speed

            new_velocity = (
                theoretical_velocity
                + self.inertia_factor * (current_velocity - theoretical_velocity)
            )

            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = new_velocity / speed * self.max_speed

        # Même discrétisation que l'environnement.
        new_velocity = np.where(
            new_velocity < 0,
            np.ceil(new_velocity),
            np.floor(new_velocity),
        ).astype(np.int32)

        return new_velocity.astype(float)

    def _sailing_efficiency(
        self,
        boat_direction: np.ndarray,
        wind_direction: np.ndarray,
    ) -> float:
        # Même logique que sailing_physics.calculate_sailing_efficiency,
        # recopiée ici pour garder l'agent autonome à la soumission.
        wind_from = -wind_direction

        wind_angle = np.arccos(
            np.clip(np.dot(wind_from, boat_direction), -1.0, 1.0)
        )

        if wind_angle < np.pi / 4:
            return 0.05

        if wind_angle < np.pi / 2:
            return 0.5 + 0.5 * (wind_angle - np.pi / 4) / (np.pi / 4)

        if wind_angle < 3 * np.pi / 4:
            return 1.0

        efficiency = 1.0 - 0.5 * (wind_angle - 3 * np.pi / 4) / (np.pi / 4)
        return max(0.5, efficiency)

    def _extract_world_map(self, observation: np.ndarray) -> np.ndarray:
        # Observation :
        # [x, y, vx, vy, wx, wy, flattened wind field, flattened world]
        world_size = self.grid_size[0] * self.grid_size[1]
        world_flat = observation[-world_size:]
        return world_flat.reshape((self.grid_size[1], self.grid_size[0]))

    def _is_unsafe(self, position: np.ndarray, world_map: np.ndarray) -> bool:
        x = int(round(position[0]))
        y = int(round(position[1]))

        if x < 0 or x >= 128 or y < 0 or y >= 128:
            return True

        # Check cellule centrale + petit voisinage.
        radius = 1
        x_min = max(0, x - radius)
        x_max = min(127, x + radius)
        y_min = max(0, y - radius)
        y_max = min(127, y + radius)

        neighborhood = world_map[y_min:y_max + 1, x_min:x_max + 1]
        return np.any(neighborhood > 0.5)

    def reset(self) -> None:
        pass

    def seed(self, seed=None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass