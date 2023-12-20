import random
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results, load_results
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import random
from pandas import Timestamp
import warnings

from flight_scheduling import FlightSchedulingEnv
from utils import generate_random_flight_schedule, generate_lambdas

warnings.filterwarnings('ignore')

random_schedule = generate_random_flight_schedule(10)
random_lambdas = generate_lambdas(random_schedule)

env = FlightSchedulingEnv(
    flight_schedule=random_schedule,
    lambdas=random_lambdas,
    max_steps=1000,
    revenue_estimation='classic'
)
env = NormalizeObservation(env)
#env = NormalizeReward(env)

model = PPO("MlpPolicy", env).learn(total_timesteps=10)

for _ in range(1):
    obs, _ = env.reset()
    env.renderer()
    total_reward = 0
    done = False
    count = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        #print('action : ', action)
        #print('obs : ', obs)
        #print('reward : ', reward)
        total_reward += reward
        env.renderer(init=False)
    print(total_reward)
env.close()
