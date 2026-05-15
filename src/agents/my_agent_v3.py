import base64
import io
import numpy as np

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------
# V3 = Q-learning tabulaire discretisé + fallback heuristique V2.
# Notions de cours utilisées :
# - MDP : état, action, transition, reward.
# - Q-value : Q(s, a) estime la qualité de choisir l'action a en état s.
# - Politique greedy à l'évaluation : on choisit argmax_a Q(s, a).
# - Discrétisation / state aggregation : on transforme un état continu en état discret.
# - Fallback heuristique : si l'état discret n'a pas été assez appris, on utilise V2.
# ---------------------------------------------------------------------

Q_TABLE_B64 = ""
N_TABLE_B64 = ""


class MyAgent(BaseAgent):
    """
    Agent V3 hybride :
    - utilise une Q-table apprise hors-ligne si disponible ;
    - sinon utilise l'heuristique V2, qui a déjà un bon score ;
    - garde des règles de sécurité contre l'île.
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

        self.q_table = self._load_table(Q_TABLE_B64)
        self.n_table = self._load_table(N_TABLE_B64)

        self.min_visits = 3
        self.last_positions = []

    def act(self, observation: np.ndarray) -> int:
        position = np.array([observation[0], observation[1]], dtype=float)
        world_map = self._extract_world_map(observation)

        safe_actions = self._safe_actions(observation, world_map)
        if not safe_actions:
            return int(self._heuristic_action(observation))

        q_action = self._q_action(observation, safe_actions)

        if q_action is not None:
            action = q_action
        else:
            action = self._heuristic_action(observation)

        if action not in safe_actions:
            action = self._best_safe_heuristic_action(observation, safe_actions)

        self._update_memory(position)

        if self._seems_blocked():
            action = self._anti_block_action(observation, safe_actions)

        return int(action)

    def _q_action(self, observation, safe_actions):
        key = self._state_key(observation)

        if key not in self.q_table:
            return None

        q_values = self.q_table[key]
        visits = self.n_table.get(key, np.zeros(9))

        candidate_actions = []
        for action in safe_actions:
            if visits[action] >= self.min_visits:
                candidate_actions.append(action)

        if not candidate_actions:
            return None

        best_action = max(candidate_actions, key=lambda a: q_values[a])
        return int(best_action)

    def _state_key(self, observation: np.ndarray) -> tuple:
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        x_bin = int(np.clip(x // 4, 0, 31))
        y_bin = int(np.clip(y // 4, 0, 31))

        vx_bin = int(np.clip(np.round(vx), -8, 8))
        vy_bin = int(np.clip(np.round(vy), -8, 8))

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 8) % 8

        if y < 18:
            phase = 0
        elif y < 90:
            phase = 1
        else:
            phase = 2

        if 35 <= x <= 93 and 15 <= y <= 88:
            danger_zone = 1
        else:
            danger_zone = 0

        return (x_bin, y_bin, vx_bin, vy_bin, wind_bin, phase, danger_zone)

    def _heuristic_action(self, observation: np.ndarray) -> int:
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

        for action in range(8):
            direction = self.directions[action]
            new_velocity = self._predict_velocity(velocity, wind, direction)
            new_position = np.clip(position + new_velocity, [0, 0], [127, 127])

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

    def _best_safe_heuristic_action(self, observation: np.ndarray, safe_actions: list[int]) -> int:
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        position = np.array([x, y], dtype=float)
        velocity = np.array([vx, vy], dtype=float)
        wind = np.array([wx, wy], dtype=float)
        world_map = self._extract_world_map(observation)
        target = self._choose_target(position)

        best_action = safe_actions[0]
        best_score = -1e18

        for action in safe_actions:
            direction = self.directions[action]
            new_velocity = self._predict_velocity(velocity, wind, direction)
            new_position = np.clip(position + new_velocity, [0, 0], [127, 127])

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

    def _safe_actions(self, observation: np.ndarray, world_map: np.ndarray) -> list[int]:
        position = np.array([observation[0], observation[1]], dtype=float)
        velocity = np.array([observation[2], observation[3]], dtype=float)
        wind = np.array([observation[4], observation[5]], dtype=float)

        safe = []

        for action in range(8):
            direction = self.directions[action]
            new_velocity = self._predict_velocity(velocity, wind, direction)
            new_position = np.clip(position + new_velocity, [0, 0], [127, 127])

            if not self._is_unsafe(new_position, world_map):
                safe.append(action)

        return safe

    def _choose_target(self, position: np.ndarray) -> np.ndarray:
        x, y = position

        if y < 18:
            return np.array([31.0, 18.0])

        if y < 90:
            return np.array([31.0, 92.0])

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
        score += 0.5 * velocity[1]

        if velocity[1] < 0:
            score -= 5.0

        if self._is_unsafe(new_position, world_map):
            score -= 10_000.0

        x, y = new_position
        if 35 <= x <= 93 and 15 <= y <= 88:
            score -= 300.0

        if action == 8:
            score -= 100.0

        return score

    def _anti_block_action(self, observation: np.ndarray, safe_actions: list[int]) -> int:
        if not safe_actions:
            return int(self._heuristic_action(observation))

        y = observation[1]

        if y < 90:
            preferred = [7, 0, 1, 6, 2]
        else:
            preferred = [1, 0, 7, 2, 6]

        for action in preferred:
            if action in safe_actions:
                return int(action)

        return int(safe_actions[0])

    def _update_memory(self, position: np.ndarray) -> None:
        self.last_positions.append(position.copy())
        if len(self.last_positions) > 12:
            self.last_positions.pop(0)

    def _seems_blocked(self) -> bool:
        if len(self.last_positions) < 12:
            return False

        displacement = np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        return displacement < 5.0

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

            theoretical_velocity = direction * efficiency * wind_norm * self.boat_performance

            speed = np.linalg.norm(theoretical_velocity)
            if speed > self.max_speed:
                theoretical_velocity = theoretical_velocity / speed * self.max_speed

            new_velocity = theoretical_velocity + self.inertia_factor * (
                current_velocity - theoretical_velocity
            )

            speed = np.linalg.norm(new_velocity)
            if speed > self.max_speed:
                new_velocity = new_velocity / speed * self.max_speed

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
        world_size = self.grid_size[0] * self.grid_size[1]
        world_flat = observation[-world_size:]
        return world_flat.reshape((self.grid_size[1], self.grid_size[0]))

    def _is_unsafe(self, position: np.ndarray, world_map: np.ndarray) -> bool:
        x = int(round(position[0]))
        y = int(round(position[1]))

        if x < 0 or x >= 128 or y < 0 or y >= 128:
            return True

        radius = 1
        x_min = max(0, x - radius)
        x_max = min(127, x + radius)
        y_min = max(0, y - radius)
        y_max = min(127, y + radius)

        neighborhood = world_map[y_min:y_max + 1, x_min:x_max + 1]
        return bool(np.any(neighborhood > 0.5))

    def _load_table(self, table_b64: str) -> dict:
        if not table_b64:
            return {}

        raw = base64.b64decode(table_b64.encode("ascii"))
        buffer = io.BytesIO(raw)
        data = np.load(buffer, allow_pickle=True)

        keys = data["keys"]
        values = data["values"]

        table = {}
        for key, value in zip(keys, values):
            table[tuple(int(v) for v in key)] = value.astype(float)

        return table

    def reset(self) -> None:
        self.last_positions = []

    def seed(self, seed=None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass