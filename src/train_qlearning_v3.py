import argparse
import base64
import io
import os
import sys
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------
# Entraînement V3 par Q-learning tabulaire.
# Notions de cours utilisées :
# - Interaction séquentielle avec simulateur : on collecte (s, a, r, s').
# - Epsilon-greedy : exploration au début, exploitation progressivement.
# - Update Q-learning off-policy : cible r + gamma * max_a' Q(s', a').
# - Reward shaping : on ajoute un signal dense pour aider l'apprentissage.
# - Export : la Q-table est encodée directement dans my_agent.py pour Codabench.
# ---------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

for path in [ROOT, SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from src.agents.my_agent_v3 import MyAgent

def make_env(scenario_name: str, max_horizon: int = 500) -> SailingEnv:
    params = get_wind_scenario(scenario_name)
    return SailingEnv(
        **params,
        max_horizon=max_horizon,
        render_mode=None,
        show_full_trajectory=False,
    )


def state_key(observation: np.ndarray) -> tuple:
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


def shaped_reward(
    env_reward: float,
    old_observation: np.ndarray,
    new_observation: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> float:
    goal = np.array([64.0, 127.0])

    old_pos = np.array([old_observation[0], old_observation[1]], dtype=float)
    new_pos = np.array([new_observation[0], new_observation[1]], dtype=float)

    old_dist = np.linalg.norm(goal - old_pos)
    new_dist = np.linalg.norm(goal - new_pos)

    progress = old_dist - new_dist

    reward = 0.0
    reward += env_reward
    reward += 1.0 * progress
    reward -= 0.03

    x, y = new_pos
    if 35 <= x <= 93 and 15 <= y <= 88:
        reward -= 4.0

    if info.get("is_stuck", False):
        reward -= 100.0

    if truncated:
        reward -= 20.0

    return float(reward)


def choose_action(
    observation: np.ndarray,
    q_table: dict,
    n_table: dict,
    helper_agent: MyAgent,
    epsilon: float,
) -> int:
    key = state_key(observation)

    if np.random.random() < epsilon:
        world_map = helper_agent._extract_world_map(observation)
        safe_actions = helper_agent._safe_actions(observation, world_map)

        if safe_actions:
            return int(np.random.choice(safe_actions))

        return int(np.random.randint(0, 8))

    if key in q_table and np.max(n_table[key]) > 0:
        world_map = helper_agent._extract_world_map(observation)
        safe_actions = helper_agent._safe_actions(observation, world_map)

        if safe_actions:
            return int(max(safe_actions, key=lambda a: q_table[key][a]))

        return int(np.argmax(q_table[key]))

    return int(helper_agent.act(observation))


def train_qlearning(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    seed_start: int,
) -> tuple[dict, dict]:
    q_table = defaultdict(lambda: np.zeros(9, dtype=float))
    n_table = defaultdict(lambda: np.zeros(9, dtype=float))

    scenarios = ["training_1", "training_2", "training_3"]
    helper_agent = MyAgent()

    global_episode = 0

    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        env = make_env(scenario)

        seed = seed_start + episode
        observation, _ = env.reset(seed=seed)
        helper_agent.reset()
        helper_agent.seed(seed)

        ratio = episode / max(1, episodes - 1)
        epsilon = epsilon_start * (1 - ratio) + epsilon_end * ratio

        total_env_reward = 0.0
        total_shaped_reward = 0.0
        success = False

        for step in range(500):
            s = state_key(observation)
            action = choose_action(
                observation=observation,
                q_table=q_table,
                n_table=n_table,
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
            n_table[s][action] += 1

            observation = new_observation
            total_env_reward += env_reward
            total_shaped_reward += r

            if terminated or truncated:
                success = env_reward > 0
                break

        global_episode += 1

        if global_episode % 100 == 0:
            print(
                f"episode={global_episode:5d} | "
                f"scenario={scenario} | "
                f"epsilon={epsilon:.3f} | "
                f"success={success} | "
                f"steps={step + 1:3d} | "
                f"env_reward={total_env_reward:.1f} | "
                f"shaped={total_shaped_reward:.1f} | "
                f"states={len(q_table)}"
            )

    return dict(q_table), dict(n_table)


def encode_table(table: dict) -> str:
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


def export_agent(q_table: dict, n_table: dict, output_path: str) -> None:
    agent_path = os.path.join(SRC, "agents", "my_agent_v3.py")

    with open(agent_path, "r", encoding="utf-8") as f:
        code = f.read()

    q_b64 = encode_table(q_table)
    n_b64 = encode_table(n_table)

    code = code.replace('Q_TABLE_B64 = ""', f'Q_TABLE_B64 = "{q_b64}"')
    code = code.replace('N_TABLE_B64 = ""', f'N_TABLE_B64 = "{n_b64}"')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"\nAgent exporté vers : {output_path}")
    print(f"Nombre d'états appris : {len(q_table)}")
    print(f"Taille du fichier : {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "submission_v3", "my_agent.py"),
    )

    args = parser.parse_args()

    q_table, n_table = train_qlearning(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed_start=args.seed_start,
    )

    export_agent(q_table, n_table, args.output)


if __name__ == "__main__":
    main()