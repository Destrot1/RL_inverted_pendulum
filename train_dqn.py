import os
import gymnasium as gym #for the physical simulation environment creation
from stable_baselines3 import DQN
import imageio

def main():
    # environment
    env = gym.make('CartPole-v1', render_mode='rgb_array')  # Discrete action space [move_left, move_right]
    # env = gym.make('CartPole-v1', render_mode='human') # GUI

    # agent
    agent = DQN(
        'MlpPolicy',             # Multi-Layer Perceptron policy
        env,
        learning_rate=1e-3,
        buffer_size=int(5e4),      # Replay Buffer size
        learning_starts=int(1e3),   # Agent starts learning after 1000 random steps
        batch_size=32,
        tau=1.0,                 # Target_net update interpolation
        gamma=0.99,              # Discount factor for future rewards
        train_freq=4,            # Frequency (in steps) of model updates
        target_update_interval=1e3,  # Steps before Target_net is updated
        verbose=1,               # Logging level
        tensorboard_log='./logs/dqn_cartpole_logs/'  # TensorBoard log directory
    )

    # training
    TIMESTEPS = int(1e5)
    agent.learn(total_timesteps=TIMESTEPS, log_interval=10)

    # saving trained agent
    os.makedirs('agents', exist_ok=True)
    agent.save('agents/dqn_agent')
    print("Agent trained and saved\n")

    # test trained agent and save GIFs
    os.makedirs('results', exist_ok=True)

    # env = gym.make('CartPole-v1', render_mode='human') # GUI
    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()  # Reset environment (returns initial state)
        done = False
        total_reward = 0
        frames = [] # frame list for GIF

        while not done:
            frame = env.render()
            frames.append(frame)

            # Agent selects best action based on the current observation
            action, _ = agent.predict(obs, deterministic=True)

            # Environment executes the action and returns new state & reward
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode finished
            done = terminated or truncated

            # Accumulate total reward
            total_reward += reward

        # save GIF
        gif_path = f'results/episode_{ep+1}.gif'
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Episode {ep + 1}: total reward = {total_reward}, GIF saved to {gif_path}")

    # --- Close environment ---
    env.close()


if __name__ == "__main__":
    main()