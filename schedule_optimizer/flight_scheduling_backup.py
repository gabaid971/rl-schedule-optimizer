from gymnasium import spaces
import gymnasium as gym
from matplotlib.animation import FuncAnimation
import numpy as np
from schedule_optimizer.revenue import Revenue
from schedule_optimizer.utils import transform_schedule
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Cache polars import to avoid repeated imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

# Cache polars import to avoid repeated imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None
import time

class FlightSchedulingEnv(gym.Env):
    def __init__(self, flight_schedule, lambdas, max_steps=100, revenue_estimation='basic', obs='', obs_back="numpy"):
        self.number_of_flights = len(flight_schedule)
        self.obs = obs
        self.flight_schedule = flight_schedule
        self.lambdas = lambdas
        self.revenue_estimation = revenue_estimation
        self.obs_back = obs_back
        
        # Timing instrumentation
        self.timers = {
            'revenue_calculation': 0.0,
            'update_current': 0.0,
            'get_observations': 0.0,
            'step_total': 0.0,
            'reset_total': 0.0
        }
        self.step_count = 0
        
        # Revenue caching for optimization
        self.cached_revenue = None

        self.action_space = spaces.Discrete(self.number_of_flights*2 + 1) 

        self.flight_schedule_minutes = flight_schedule[
            ['departure_minutes', 'arrival_minutes']
        ]
        self.connections = transform_schedule(self.flight_schedule)

        if self.obs_back == "polars":
            self.current = {
                'flight_schedule': np.array(self.flight_schedule_minutes),
                'connections': self.connections
            }
        else:
            self.current = {
                'flight_schedule': np.array(self.flight_schedule_minutes),
                'connections': np.array(self.connections)
            }
            
        if self.obs == 'flights':
            self.observation_space = spaces.Box(
                low = 0,
                high = 3000,
                shape = (self.number_of_flights, 2),
                dtype = np.float32
            )
        else:
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

    def calculate_preference_value(self, cnx_time):
        if 60 <= cnx_time <= 120:
            return 0.5 + 0.5 * (cnx_time - 60) / 60
        elif 120 < cnx_time < 480:
            return 1 - 0.5 * (cnx_time - 120) / 360
        else:
            return 0
    
    def get_observations(self):
        obs_start = time.perf_counter()
        
        if self.obs == 'flights':
            observations = np.float32(self.current['flight_schedule'])
        elif self.obs == "cnx":
            # support Polars DataFrame or numpy array
            conn = self.current['connections']
            try:
                import polars as pl
                if isinstance(conn, pl.DataFrame):
                    observations = np.float32(conn['cnx_time'].to_numpy())
                else:
                    observations = np.float32(conn[:, 4])
            except Exception:
                # fallback
                observations = np.float32(conn[:, 4])
        elif self.obs == 'revenue_aware':  # ← NOUVEAU
            conn = self.current['connections']
            cnx_times = np.float32(conn['cnx_time'].to_numpy()) if hasattr(conn, 'to_numpy') else np.float32(conn[:, 4])
            
            revenue_features = []
            for cnx_time in cnx_times:
                pref_value = self.calculate_preference_value(cnx_time)
                revenue_features.append(pref_value)
            
            # Combiner temps connexion + preference values
            observations = np.concatenate([cnx_times, np.array(revenue_features, dtype=np.float32)])        
        obs_end = time.perf_counter()
        self.timers['get_observations'] += obs_end - obs_start
        
        return observations
    
    def reset(self, seed = None, options = None):
        reset_start = time.perf_counter()
        
        super().reset(seed=seed)
        self.current = {
            'flight_schedule': np.array(self.flight_schedule_minutes),
            'connections': self.connections
        }
        self.current_step = 0
        
        obs = self.get_observations()
        
        # Initialize revenue cache
        self.cached_revenue = self.calculate_revenue()
        
        reset_end = time.perf_counter()
        self.timers['reset_total'] += reset_end - reset_start
        
        return obs, {}

    def step(self, action):
        step_start = time.perf_counter()
        
        # Use cached revenue instead of recalculating
        old_revenue = self.cached_revenue
    
        self.update_current(action)
        # Calculate only the new revenue
        new_revenue = self.calculate_revenue()
        # Update cache for next step
        self.cached_revenue = new_revenue
        
        reward = new_revenue - old_revenue

        done = self.current_step == self.max_steps
        self.current_step += 1
        
        obs = self.get_observations()
        
        step_end = time.perf_counter()
        self.timers['step_total'] += step_end - step_start
        self.step_count += 1
        
        return obs, reward, done, False, {}
        
    def calculate_revenue(self):
        revenue_start = time.perf_counter()
        
        if self.revenue_estimation == 'basic':
            result = float(np.sum(self.current['flight_schedule'][:, 0]))
        elif self.revenue_estimation == 'classic':
            rev = Revenue(self.current, self.lambdas)
            rev_tot = rev.calculate_revenue()
            result = rev_tot
        else:
            result = float(0)
            
        revenue_end = time.perf_counter()
        self.timers['revenue_calculation'] += revenue_end - revenue_start
        
        return result

    def respect_constraints(self, flight_number, departure_time): 
        min_departure, max_departure = self.constraints[flight_number]
        return min_departure <= departure_time <= max_departure
    
    def get_timing_report(self):
        """Get a detailed timing report of environment performance"""
        if self.step_count == 0:
            return "No steps completed yet."
            
        report = f"\n=== FlightSchedulingEnv Timing Report ===\n"
        report += f"Total steps: {self.step_count}\n"
        report += f"Average step time: {self.timers['step_total'] / self.step_count * 1000:.2f} ms\n\n"
        
        for component, total_time in self.timers.items():
            if component != 'step_total':
                avg_time = total_time / max(self.step_count, 1) * 1000
                pct_of_step = (total_time / self.timers['step_total'] * 100) if self.timers['step_total'] > 0 else 0
                report += f"{component:20}: {avg_time:6.2f} ms/step ({pct_of_step:5.1f}% of step)\n"
        
        return report
        
    def reset_timers(self):
        """Reset all timing counters"""
        self.timers = {k: 0.0 for k in self.timers}
        self.step_count = 0
    
    def init_animation(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.rectangles = []

        for i in range(self.number_of_flights):
            departure_time = self.current['flight_schedule'][i, 0]
            arrival_time = self.current['flight_schedule'][i, 1]
            # Récupérer le 'way' depuis le DataFrame polars
            way = self.flight_schedule.row(i, named=True)['way']
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


    def update_current(self, action):
        update_start = time.perf_counter()
        
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
                conn = self.current['connections']
                shift = sign * self.time_step
                try:
                    import polars as pl
                    if isinstance(conn, pl.DataFrame):
                        # compute delta: -shift where flight_number_1 == flight_name, +shift where flight_number_2 == flight_name
                        delta = (
                            pl.when(pl.col('flight_number_1') == flight_name).then(-shift).otherwise(0)
                            + pl.when(pl.col('flight_number_2') == flight_name).then(shift).otherwise(0)
                        )
                        conn = conn.with_columns((pl.col('cnx_time') + delta).alias('cnx_time'))
                        self.current['connections'] = conn
                        return
                except Exception:
                    # polars not available or error; fall back to numpy below
                    pass

                # Fallback if connections is numpy array-like
                try:
                    self.current['connections'][self.current['connections'][:, 0] == flight_name, 4] -= shift
                    self.current['connections'][self.current['connections'][:, 1] == flight_name, 4] += shift
                except Exception:
                    # If shape/types are unexpected, ignore update (defensive)
                    pass
                    
        update_end = time.perf_counter()
        self.timers['update_current'] += update_end - update_start

    def get_observations(self):
        obs_start = time.perf_counter()
        
        if self.obs == 'flights':
            observations = np.float32(self.current['flight_schedule'])
        else:
            # support Polars DataFrame or numpy array
            conn = self.current['connections']
            try:
                import polars as pl
                if isinstance(conn, pl.DataFrame):
                    observations = np.float32(conn['cnx_time'].to_numpy())
                else:
                    observations = np.float32(conn[:, 4])
            except Exception:
                # fallback
                observations = np.float32(conn[:, 4])
                
        obs_end = time.perf_counter()
        self.timers['get_observations'] += obs_end - obs_start
        
        return observations
    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        # Préserver le type original des connections (Polars vs NumPy selon obs_back)
        if self.obs_back == "polars":
            self.current = {
                'flight_schedule': np.array(self.flight_schedule_minutes),
                'connections': self.connections  # Préserver DataFrame Polars
            }
        else:
            self.current = {
                'flight_schedule': np.array(self.flight_schedule_minutes),
                'connections': np.array(self.connections)  # Convertir en NumPy
            }
        self.current_step = 0
        
        # Initialize revenue cache
        self.cached_revenue = self.calculate_revenue()
        
        return self.get_observations(), {}

    def step(self, action):
        step_start = time.perf_counter()
        
        # Use cached revenue instead of recalculating
        old_revenue = self.cached_revenue
    
        self.update_current(action)
        # Calculate only the new revenue
        new_revenue = self.calculate_revenue()
        # Update cache for next step
        self.cached_revenue = new_revenue
        
        reward = new_revenue - old_revenue

        done = self.current_step == self.max_steps
        self.current_step += 1
        
        obs = self.get_observations()
        
        step_end = time.perf_counter()
        self.timers['step_total'] += step_end - step_start
        self.step_count += 1
        
        return obs, reward, done, False, {}
        
    def calculate_revenue(self):
        revenue_start = time.perf_counter()
        
        if self.revenue_estimation == 'basic':
            result = float(np.sum(self.current['flight_schedule'][:, 0]))
        elif self.revenue_estimation == 'classic':
            rev = Revenue(self.current, self.lambdas)
            rev_tot = rev.calculate_revenue()
            result = rev_tot
        else:
            result = float(0)
            
        revenue_end = time.perf_counter()
        self.timers['revenue_calculation'] += revenue_end - revenue_start
        
        return result

    def respect_constraints(self, flight_number, departure_time): 
        min_departure, max_departure = self.constraints[flight_number]
        return min_departure <= departure_time <= max_departure
    
    def init_animation(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        self.rectangles = []

        for i in range(self.number_of_flights):
            departure_time = self.current['flight_schedule'][i, 0]
            arrival_time = self.current['flight_schedule'][i, 1]
            # Récupérer le 'way' depuis le DataFrame polars
            way = self.flight_schedule.row(i, named=True)['way']
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