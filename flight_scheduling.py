from gymnasium import spaces
import gymnasium as gym
import numpy as np

from revenue import Revenue

class FlightSchedulingEnv(gym.Env):
    def __init__(self, flight_schedule, lambdas, max_steps=100, revenue_estimation='basic'):
        self.number_of_flights = len(flight_schedule)
        self.flight_schedule = flight_schedule
        self.lambdas = lambdas
        self.revenue_estimation = revenue_estimation

        self.action_space = spaces.Discrete(self.number_of_flights*2 + 1) 
        self.observation_space = spaces.Box(low=0, 
            high=3000, 
            shape=(self.number_of_flights, 2), 
            dtype=np.int64
        )
        self.observation_space = spaces.Dict({
            'time_data': self.observation_space,
            'way': spaces.MultiBinary(self.number_of_flights)
        })
        #self.flight_schedule_minutes = flight_schedule[
        #    ['departure_minutes', 'arrival_minutes', 'way_transformed']
        #]
        self.flight_schedule_minutes = flight_schedule[
            ['departure_minutes', 'arrival_minutes']
        ]
        self.way_transformed = flight_schedule['way_transformed']
        self.current_state = {
            'time_data': np.array(self.flight_schedule_minutes, dtype=np.int64),
            'way': np.array(self.way_transformed, dtype=np.int8)
        }       

        self.constraints = {i: [0, 2000] for i in range(self.number_of_flights)}

        self.time_step = 20  
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.current_state = {
            'time_data': np.array(self.flight_schedule_minutes, dtype=np.int64),
            'way': np.array(self.way_transformed, dtype=np.int8)
        }   
        self.current_step = 0
        return self.current_state, {}

    def step(self, action):
        old_revenue = self.calculate_revenue()
        if action == self.number_of_flights * 2:
            pass
        else:
            if action % 2 == 0:
                new_departure_time_minutes = self.current_state['time_data'][action // 2, 0] + self.time_step
                new_arrival_time_minutes = self.current_state['time_data'][action // 2, 1] + self.time_step
            else:
                new_departure_time_minutes = self.current_state['time_data'][action // 2, 0] - self.time_step
                new_arrival_time_minutes = self.current_state['time_data'][action // 2, 1] - self.time_step

            if self.respect_constraints(action // 2, new_departure_time_minutes):
                self.current_state['time_data'][action // 2, 0] = new_departure_time_minutes
                self.current_state['time_data'][action // 2, 1] = new_arrival_time_minutes
                
        new_revenue = self.calculate_revenue()
        reward = new_revenue - old_revenue

        done = self.current_step == self.max_steps
        self.current_step += 1
        return self.current_state.copy(), reward, done, False, {}
        
    def calculate_revenue(self):
        if self.revenue_estimation == 'basic':
            return float(np.sum(self.current_state['time_data'][:, 0]))
        elif self.revenue_estimation == 'classic':
            rev = Revenue(self.current_state, self.lambdas)
            rev_tot = rev.calculate_revenue()
            return rev_tot
        else:
            return float(0)

    def respect_constraints(self, flight_number, departure_time): 
        min_departure, max_departure = self.constraints[flight_number]
        return min_departure <= departure_time <= max_departure