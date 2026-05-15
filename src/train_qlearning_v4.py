import argparse
import base64
import io
import os
import sys
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------
# Entraînement V4 par Q-learning 100% RL.
# Notions de cours utilisées :
# - Interaction avec simulateur : trajectoires (s, a, r, s').
# - Q-learning off-policy : update vers r + gamma * max_a' Q(s',a').
# - Epsilon-greedy : compromis exploration / exploitation.
# - Reward shaping : signal dense pour apprendre malgré reward sparse.
# - State aggregation : tables exactes, grossières, par phase, globales.
# ---------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

for path in [ROOT, SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from src.agents.my_agent_v4 import MyAgent


def make_env(scenario_name, max_horizon=500):
    params = get_wind_scenario(scenario_name)
    return SailingEnv(
        **params,
        max_horizon=max_horizon,
        render_mode=None,
        show_full_trajectory=False,
    )


def state_key(observation):
    x, y = observation[0], observation[1]
    vx, vy = observation[2], observation[3]
    wx, wy = observation[4], observation[5]

    x_bin = int(np.clip(x // 4, 0, 31))
    y_bin = int(np.clip(y // 4, 0, 31))

    vx_bin = int(np.clip(np.round(vx), -8, 8))
    vy_bin = int(np.clip(np.round(vy), -8, 8))

    wind_angle = np.arctan2(wy, wx)
    wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 8) % 8

    phase = phase_from_y(y)
    zone = zone_from_position(x, y)

    return (x_bin, y_bin, vx_bin, vy_bin, wind_bin, phase, zone)


def coarse_state_key(observation):
    x, y = observation[0], observation[1]
    vx, vy = observation[2], observation[3]
    wx, wy = observation[4], observation[5]

    x_bin = int(np.clip(x // 8, 0, 15))
    y_bin = int(np.clip(y // 8, 0, 15))

    vx_bin = int(np.clip(np.round(vx / 2), -4, 4))
    vy_bin = int(np.clip(np.round(vy / 2), -4, 4))

    wind_angle = np.arctan2(wy, wx)
    wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 4) % 4

    phase = phase_from_y(y)
    zone = zone_from_position(x, y)

    return (x_bin, y_bin, vx_bin, vy_bin, wind_bin, phase, zone)


def phase_key(observation):
    y = observation[1]
    wx, wy = observation[4], observation[5]

    wind_angle = np.arctan2(wy, wx)
    wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * 8) % 8

    return (phase_from_y(y), wind_bin)


def phase_from_y(y):
    if y < 18:
        return 0
    if y < 90:
        return 1
    return 2


def zone_from_position(x, y):
    if 35 <= x <= 93 and 15 <= y <= 88:
        return 2
    if 25 <= x <= 103 and 8 <= y <= 98:
        return 1
    return 0


def distance_to_goal(observation):
    goal = np.array([64.0, 127.0])
    position = np.array([observation[0], observation[1]], dtype=float)
    return float(np.linalg.norm(goal - position))


def shaped_reward(env_reward, old_observation, new_observation, terminated, truncated, info):
    old_dist = distance_to_goal(old_observation)
    new_dist = distance_to_goal(new_observation)

    progress = old_dist - new_dist

    old_y = old_observation[1]
    new_y = new_observation[1]

    reward = 0.0

    reward += float(env_reward)

    reward += 2.0 * progress
    reward += 0.15 * (new_y - old_y)

    reward -= 0.04

    x, y = new_observation[0], new_observation[1]

    if 35 <= x <= 93 and 15 <= y <= 88:
        reward -= 15.0

    if 25 <= x <= 103 and 8 <= y <= 98:
        reward -= 1.5

    if info.get("is_stuck", False):
        reward -= 100.0

    if truncated:
        reward -= 30.0

    if terminated and env_reward > 0:
        reward += 50.0

    return float(reward)


def choose_action(observation, q_table, coarse_q_table, phase_q_table, global_q, helper_agent, epsilon):
    world_map = helper_agent._extract_world_map(observation)
    safe_actions = helper_agent._safe_actions(observation, world_map)

    if not safe_actions:
        safe_actions = list(range(8))

    if np.random.random() < epsilon:
        return int(np.random.choice(safe_actions))

    exact_key = state_key(observation)

    if exact_key in q_table and np.max(np.abs(q_table[exact_key])) > 1e-12:
        return argmax_over_actions(q_table[exact_key], safe_actions)

    c_key = coarse_state_key(observation)

    if c_key in coarse_q_table and np.max(np.abs(coarse_q_table[c_key])) > 1e-12:
        return argmax_over_actions(coarse_q_table[c_key], safe_actions)

    p_key = phase_key(observation)

    if p_key in phase_q_table and np.max(np.abs(phase_q_table[p_key])) > 1e-12:
        return argmax_over_actions(phase_q_table[p_key], safe_actions)

    return argmax_over_actions(global_q, safe_actions)


def argmax_over_actions(q_values, actions):
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


def update_average_table(table, count_table, key, action, target):
    count_table[key][action] += 1.0
    n = count_table[key][action]
    table[key][action] += (target - table[key][action]) / n


def train_qlearning(episodes, alpha, gamma, epsilon_start, epsilon_end, seed_start):
    q_table = defaultdict(lambda: np.zeros(9, dtype=float))
    n_table = defaultdict(lambda: np.zeros(9, dtype=float))

    coarse_q_table = defaultdict(lambda: np.zeros(9, dtype=float))
    coarse_n_table = defaultdict(lambda: np.zeros(9, dtype=float))

    phase_q_table = defaultdict(lambda: np.zeros(9, dtype=float))
    phase_n_table = defaultdict(lambda: np.zeros(9, dtype=float))

    global_q = np.zeros(9, dtype=float)
    global_n = np.zeros(9, dtype=float)

    scenarios = ["training_1", "training_2", "training_3"]
    helper_agent = MyAgent()

    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        env = make_env(scenario)

        seed = seed_start + episode
        observation, _ = env.reset(seed=seed)
        helper_agent.reset()
        helper_agent.seed(seed)

        ratio = episode / max(1, episodes - 1)
        epsilon = epsilon_start * (1.0 - ratio) + epsilon_end * ratio

        total_env_reward = 0.0
        total_shaped_reward = 0.0
        success = False

        for step in range(500):
            s = state_key(observation)
            s_coarse = coarse_state_key(observation)
            s_phase = phase_key(observation)

            action = choose_action(
                observation=observation,
                q_table=q_table,
                coarse_q_table=coarse_q_table,
                phase_q_table=phase_q_table,
                global_q=global_q,
                helper_agent=helper_agent,
                epsilon=epsilon,
            )

            new_observation, env_reward, terminated, truncated, info = env.step(action)

            r = shaped_reward(
                env_reward=env_reward,
                old_observation=observation,
                new_observation=new_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

            s_next = state_key(new_observation)
            td_target = r + gamma * np.max(q_table[s_next]) * (not terminated)
            td_error = td_target - q_table[s][action]

            q_table[s][action] += alpha * td_error
            n_table[s][action] += 1.0

            update_average_table(
                table=coarse_q_table,
                count_table=coarse_n_table,
                key=s_coarse,
                action=action,
                target=td_target,
            )

            update_average_table(
                table=phase_q_table,
                count_table=phase_n_table,
                key=s_phase,
                action=action,
                target=td_target,
            )

            global_n[action] += 1.0
            global_q[action] += (td_target - global_q[action]) / global_n[action]

            observation = new_observation
            total_env_reward += env_reward
            total_shaped_reward += r

            if terminated or truncated:
                success = env_reward > 0
                break

        if (episode + 1) % 100 == 0:
            print(
                f"episode={episode + 1:5d} | "
                f"scenario={scenario} | "
                f"epsilon={epsilon:.3f} | "
                f"success={success} | "
                f"steps={step + 1:3d} | "
                f"env_reward={total_env_reward:.1f} | "
                f"shaped={total_shaped_reward:.1f} | "
                f"states={len(q_table)} | "
                f"coarse={len(coarse_q_table)}"
            )

    return dict(q_table), dict(n_table), dict(coarse_q_table), dict(phase_q_table), global_q


def encode_table(table):
    keys = []
    values = []

    for key, value in table.items():
        keys.append(np.array(key, dtype=np.int16))
        values.append(np.array(value, dtype=np.float32))

    if not keys:
        keys_array = np.empty((0, 7), dtype=np.int16)
        values_array = np.empty((0, 9), dtype=np.float32)
    else:
        keys_array = np.vstack(keys)
        values_array = np.vstack(values)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, keys=keys_array, values=values_array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_vector(vector):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, values=np.array(vector, dtype=np.float32))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def export_agent(q_table, n_table, coarse_q_table, phase_q_table, global_q, output_path):
    agent_path = os.path.join(SRC, "agents", "my_agent_v4.py")

    with open(agent_path, "r", encoding="utf-8") as f:
        code = f.read()

    q_b64 = encode_table(q_table)
    n_b64 = encode_table(n_table)
    coarse_b64 = encode_table(coarse_q_table)
    phase_b64 = encode_table(phase_q_table)
    global_b64 = encode_vector(global_q)

    code = code.replace('Q_TABLE_B64 = ""', f'Q_TABLE_B64 = "{q_b64}"')
    code = code.replace('N_TABLE_B64 = ""', f'N_TABLE_B64 = "{n_b64}"')
    code = code.replace('COARSE_Q_TABLE_B64 = ""', f'COARSE_Q_TABLE_B64 = "{coarse_b64}"')
    code = code.replace('PHASE_Q_TABLE_B64 = ""', f'PHASE_Q_TABLE_B64 = "{phase_b64}"')
    code = code.replace('GLOBAL_Q_B64 = ""', f'GLOBAL_Q_B64 = "{global_b64}"')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\nAgent exporté vers : {output_path}")
    print(f"Nombre d'états exacts appris : {len(q_table)}")
    print(f"Nombre d'états grossiers appris : {len(coarse_q_table)}")
    print(f"Nombre d'états phase appris : {len(phase_q_table)}")
    print(f"Taille du fichier : {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epsilon-start", type=float, default=0.70)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "submission_v4", "my_agent.py"),
    )

    args = parser.parse_args()

    q_table, n_table, coarse_q_table, phase_q_table, global_q = train_qlearning(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed_start=args.seed_start,
    )

    export_agent(
        q_table=q_table,
        n_table=n_table,
        coarse_q_table=coarse_q_table,
        phase_q_table=phase_q_table,
        global_q=global_q,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()