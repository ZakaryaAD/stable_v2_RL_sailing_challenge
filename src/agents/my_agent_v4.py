import base64
import io
import numpy as np

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------
# V4 = agent 100% RL par Q-learning tabulaire discretisé.
# Notions de cours utilisées :
# - MDP : l'état contient position, vitesse, vent et carte ; l'action est une direction.
# - Q-value : Q(s,a) estime la valeur future d'une action dans un état.
# - Policy greedy à l'évaluation : on choisit argmax_a Q(s,a).
# - State aggregation : observation continue -> état discret compact.
# - Action masking : on ne garde que les actions physiquement sûres.
# - Fallback RL : si l'état exact est inconnu, on utilise des Q-values agrégées apprises.
# ---------------------------------------------------------------------

Q_TABLE_B64 = ""
N_TABLE_B64 = ""
COARSE_Q_TABLE_B64 = ""
PHASE_Q_TABLE_B64 = ""
GLOBAL_Q_B64 = ""


class MyAgent(BaseAgent):
    """
    Agent V4 100% RL.

    À l'évaluation :
    - pas d'apprentissage ;
    - pas de fallback heuristique ;
    - choix greedy sur Q-table si état connu ;
    - sinon choix greedy sur Q-values agrégées apprises pendant l'entraînement.
    """

    def __init__(self):
        super().__init__()

        self.grid_size = (128, 128)
        self.goal_position = np.array([64.0, 127.0])

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
        self.coarse_q_table = self._load_table(COARSE_Q_TABLE_B64)
        self.phase_q_table = self._load_table(PHASE_Q_TABLE_B64)
        self.global_q = self._load_vector(GLOBAL_Q_B64)

        self.min_visits_exact = 2
        self.min_visits_coarse = 1

        self.np_random = np.random.default_rng()
        self.last_positions = []

    def act(self, observation):
        position = np.array([observation[0], observation[1]], dtype=float)
        world_map = self._extract_world_map(observation)

        safe_actions = self._safe_actions(observation, world_map)

        if not safe_actions:
            safe_actions = list(range(8))

        action = self._rl_greedy_action(observation, safe_actions)

        self._update_memory(position)

        if self._seems_blocked():
            action = self._rl_unblock_action(observation, safe_actions)

        return int(action)

    def _rl_greedy_action(self, observation, safe_actions):
        exact_key = self._state_key(observation)

        if exact_key in self.q_table:
            q_values = self.q_table[exact_key]
            visits = self.n_table.get(exact_key, np.zeros(9))

            visited_safe = [
                action for action in safe_actions
                if visits[action] >= self.min_visits_exact
            ]

            if visited_safe:
                return self._argmax_over_actions(q_values, visited_safe)

        coarse_key = self._coarse_state_key(observation)

        if coarse_key in self.coarse_q_table:
            q_values = self.coarse_q_table[coarse_key]
            return self._argmax_over_actions(q_values, safe_actions)

        phase_key = self._phase_key(observation)

        if phase_key in self.phase_q_table:
            q_values = self.phase_q_table[phase_key]
            return self._argmax_over_actions(q_values, safe_actions)

        return self._argmax_over_actions(self.global_q, safe_actions)

    def _rl_unblock_action(self, observation, safe_actions):
        q_candidates = []

        exact_key = self._state_key(observation)
        coarse_key = self._coarse_state_key(observation)
        phase_key = self._phase_key(observation)

        if exact_key in self.q_table:
            q_candidates.append(self.q_table[exact_key])

        if coarse_key in self.coarse_q_table:
            q_candidates.append(self.coarse_q_table[coarse_key])

        if phase_key in self.phase_q_table:
            q_candidates.append(self.phase_q_table[phase_key])

        q_candidates.append(self.global_q)

        combined_q = np.mean(np.vstack(q_candidates), axis=0)

        non_stay_actions = [a for a in safe_actions if a != 8]
        if non_stay_actions:
            safe_actions = non_stay_actions

        return self._argmax_over_actions(combined_q, safe_actions)

    def _argmax_over_actions(self, q_values, actions):
        best_action = actions[0]
        best_value = -1e18

        for action in actions:
            value = float(q_values[action])

            if action == 8:
                value -= 5.0

            if value > best_value:
                best_value = value
                best_action = action

        return int(best_action)

    def _state_key(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        x_bin = int(np.clip(x // 4, 0, 31))
        y_bin = int(np.clip(y // 4, 0, 31))

        vx_bin = int(np.clip(np.round(vx), -8, 8))
        vy_bin = int(np.clip(np.round(vy), -8, 8))

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 8) % 8

        phase = self._phase_from_y(y)
        zone = self._zone_from_position(x, y)

        return (x_bin, y_bin, vx_bin, vy_bin, wind_bin, phase, zone)

    def _coarse_state_key(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        x_bin = int(np.clip(x // 8, 0, 15))
        y_bin = int(np.clip(y // 8, 0, 15))

        vx_bin = int(np.clip(np.round(vx / 2), -4, 4))
        vy_bin = int(np.clip(np.round(vy / 2), -4, 4))

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 4) % 4

        phase = self._phase_from_y(y)
        zone = self._zone_from_position(x, y)

        return (x_bin, y_bin, vx_bin, vy_bin, wind_bin, phase, zone)

    def _phase_key(self, observation):
        y = observation[1]
        wx, wy = observation[4], observation[5]

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 8) % 8
        phase = self._phase_from_y(y)

        return (phase, wind_bin)

    def _phase_from_y(self, y):
        if y < 18:
            return 0
        if y < 90:
            return 1
        return 2

    def _zone_from_position(self, x, y):
        if 35 <= x <= 93 and 15 <= y <= 88:
            return 2
        if 25 <= x <= 103 and 8 <= y <= 98:
            return 1
        return 0

    def _safe_actions(self, observation, world_map):
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

        if not safe:
            safe = [8]

        return safe

    def _predict_velocity(self, current_velocity, wind, direction):
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

    def _sailing_efficiency(self, boat_direction, wind_direction):
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

    def _extract_world_map(self, observation):
        world_size = self.grid_size[0] * self.grid_size[1]
        world_flat = observation[-world_size:]
        return world_flat.reshape((self.grid_size[1], self.grid_size[0]))

    def _is_unsafe(self, position, world_map):
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

    def _update_memory(self, position):
        self.last_positions.append(position.copy())
        if len(self.last_positions) > 12:
            self.last_positions.pop(0)

    def _seems_blocked(self):
        if len(self.last_positions) < 12:
            return False

        displacement = np.linalg.norm(self.last_positions[-1] - self.last_positions[0])
        return displacement < 4.0

    def _load_table(self, table_b64):
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

    def _load_vector(self, vector_b64):
        if not vector_b64:
            return np.zeros(9, dtype=float)

        raw = base64.b64decode(vector_b64.encode("ascii"))
        buffer = io.BytesIO(raw)
        data = np.load(buffer, allow_pickle=True)

        return data["values"].astype(float)

    def reset(self):
        self.last_positions = []

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        pass

    def load(self, path):
        pass