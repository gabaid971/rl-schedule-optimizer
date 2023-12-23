from gymnasium import spaces
import gymnasium as gym
from matplotlib.animation import FuncAnimation
import numpy as np
from schedule_optimizer.revenue import Revenue
from schedule_optimizer.utils import transform_schedule
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlightSchedulingEnv(gym.Env):
    def __init__(self, flight_schedule, lambdas, max_steps=100, revenue_estimation='basic', obs=''):
        self.number_of_flights = len(flight_schedule)
        self.obs = obs
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
        if self.obs == 'flights':
            self.observation_space = spaces.Box(
                low = 0,
                high = 3000,
                shape = (self.number_of_flights, 2),
                dtype = np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low = -3000,
                high = 3000,
                shape = (len(self.connections), ),
                dtype = np.float64
            )
        
        self.constraints = {i: [0, 2000] for i in range(self.number_of_flights)}

        self.time_step = 20  
        self.max_steps = max_steps
        self.current_step = 0

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
        if self.obs == 'flights':
            observations = np.float32(self.current['flight_schedule'])
        else:
            observations = np.float32(self.current['connections'][:, 4])
        return observations
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.current = {
            'flight_schedule': np.array(self.flight_schedule_minutes),
            'connections': np.array(self.connections)
        }
        self.current_step = 0
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
    
    def init_animation(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.rectangles = []

        for i in range(self.number_of_flights):
            departure_time = self.current['flight_schedule'][i, 0]
            arrival_time = self.current['flight_schedule'][i, 1]
            way = self.flight_schedule['way'].iloc[i]
            flight_name = f'{way}'

            rect = patches.Rectangle((departure_time, i), arrival_time - departure_time, 0.8, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(departure_time, i + 0.4, flight_name, ha='left', va='center', color='b')
            self.rectangles.append(rect)

        ax.set_xlim(0, max(self.current['flight_schedule'][:, 1]) + 100)
        ax.set_ylim(-1, self.number_of_flights)
        plt.xlabel('Temps (minutes)')
        plt.ylabel('Vols')
        plt.title('Évolution du planning de vol')
        plt.grid(True)
        
        return fig, ax

    def update_animation(self, frame):
        for i in range(self.number_of_flights):
            departure_time = self.current['flight_schedule'][i, 0]
            arrival_time = self.current['flight_schedule'][i, 1]
            self.rectangles[i].set_x(departure_time)
            self.rectangles[i].set_width(arrival_time - departure_time)

        return self.rectangles

    def renderer(self, init=True, episode_length=100):
        fig, ax = self.init_animation()
        
        if init:
            plt.show()
        else: 
            animation = FuncAnimation(fig, self.update_animation, frames=episode_length, blit=True)
            plt.show()