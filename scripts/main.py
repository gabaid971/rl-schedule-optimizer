from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation
from flight_scheduling import FlightSchedulingEnv
from utils import generate_random_flight_schedule, generate_lambdas


def main():
    random_schedule = generate_random_flight_schedule(10)
    random_lambdas = generate_lambdas(random_schedule)

    env = FlightSchedulingEnv(
        flight_schedule=random_schedule,
        lambdas=random_lambdas,
        max_steps=100,
        revenue_estimation='classic'
    )
    env = NormalizeObservation(env)

    model = PPO("MlpPolicy", env).learn(total_timesteps=100000)

    for _ in range(1):
        obs, _ = env.reset()
        env.renderer()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.renderer(init=False)
        print(total_reward)
    env.close()


if __name__ == '__main__':
    main()