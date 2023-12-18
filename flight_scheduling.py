from gymnasium import spaces
import gymnasium as gym
import numpy as np
#from sklearn.preprocessing import StandardScaler
from revenue import Revenue
from utils import transform_schedule

class FlightSchedulingEnv(gym.Env):
    def __init__(self, flight_schedule, lambdas, max_steps=100, revenue_estimation='basic'):
        self.number_of_flights = len(flight_schedule)
        self.flight_schedule = flight_schedule
        self.lambdas = lambdas
        self.revenue_estimation = revenue_estimation

        self.action_space = spaces.Discrete(self.number_of_flights*2 + 1) 

        self.flight_schedule_minutes = flight_schedule[
            ['departure_minutes', 'arrival_minutes']
        ]
        self.connections = transform_schedule(self.flight_schedule)

        self.current = {
            'flight_schedule': np.array(self.flight_schedule_minutes),
            'connections': np.array(self.connections)
        }
  
        self.observation_space = spaces.Box(
            low = -3000,
            high = 3000,
            shape = (len(self.connections), ),
            dtype = np.float32
        )
        self.constraints = {i: [0, 2000] for i in range(self.number_of_flights)}

        self.time_step = 20  
        self.max_steps = max_steps
        self.current_step = 0
        #self.scaler = StandardScaler()

    def update_current(self, action):
        flight_name = action // 2
        if action == self.number_of_flights * 2:
            pass
        else:
            sign = 1 if action % 2 else -1
            new_departure_time_minutes = self.current['flight_schedule'][flight_name, 0]\
                  + sign * self.time_step
            new_arrival_time_minutes = self.current['flight_schedule'][flight_name, 1]\
                  + sign * self.time_step

            if self.respect_constraints(flight_name, new_departure_time_minutes):
                # update schedule
                self.current['flight_schedule'][flight_name, 0] = new_departure_time_minutes
                self.current['flight_schedule'][flight_name, 1] = new_arrival_time_minutes
                # update connections
                self.current['connections'][self.current['connections'][:, 0] == flight_name, 4]\
                      -= sign * self.time_step
                self.current['connections'][self.current['connections'][:, 1] == flight_name, 4]\
                      += sign * self.time_step

    def get_observations(self):
        observations = np.float32(self.current['connections'][:, 4])
        #observations = self.scaler.transform(observations.reshape(-1, 1)).flatten()
        return observations
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.current = {
            'flight_schedule': np.array(self.flight_schedule_minutes),
            'connections': np.array(self.connections)
        }
        self.current_step = 0
        #self.scaler.fit(self.get_observations().reshape(-1, 1))
        return self.get_observations(), {}

    def step(self, action):
        old_revenue = self.calculate_revenue()
    
        self.update_current(action)
        new_revenue = self.calculate_revenue()
        reward = new_revenue - old_revenue

        done = self.current_step == self.max_steps
        self.current_step += 1
        return self.get_observations(), reward, done, False, {}
        
    def calculate_revenue(self):
        if self.revenue_estimation == 'basic':
            return float(np.sum(self.current['flight_schedule'][:, 0]))
        elif self.revenue_estimation == 'classic':
            rev = Revenue(self.current, self.lambdas)
            rev_tot = rev.calculate_revenue()
            return rev_tot
        else:
            return float(0)

    def respect_constraints(self, flight_number, departure_time): 
        min_departure, max_departure = self.constraints[flight_number]
        return min_departure <= departure_time <= max_departure