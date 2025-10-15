from schedule_optimizer.flight_scheduling import FlightSchedulingEnv
import numpy as np


class FlightSchedulingEnvMasked(FlightSchedulingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = {i: [0, 2000] for i in range(self.number_of_flights)}
        
    def get_action_mask(self):
        """Retourne un masque boolean indiquant quelles actions sont valides"""
        mask = np.ones(self.action_space.n, dtype=bool)
        
        for flight_idx in range(self.number_of_flights):
            # Action increase (idx * 2 + 1)
            increase_action_idx = flight_idx * 2 + 1
            current_departure = self.current['flight_schedule'][flight_idx, 0]
            new_departure_increase = current_departure + self.time_step
            
            # Action decrease (idx * 2)
            decrease_action_idx = flight_idx * 2
            new_departure_decrease = current_departure - self.time_step
            
            # Vérifier les contraintes
            if not self.respect_constraints(flight_idx, new_departure_increase):
                mask[increase_action_idx] = False
                
            if not self.respect_constraints(flight_idx, new_departure_decrease):
                mask[decrease_action_idx] = False
        
        # L'action "no-op" (dernière action) est toujours valide
        mask[-1] = True
        
        return mask
    
    def update_current(self, action):
        """Version optimisée qui fait confiance à ActionMasker"""
        import time
        update_start = time.perf_counter()
        
        flight_name = action // 2
        if action == self.number_of_flights * 2:
            # Action no-op
            pass
        else:
            sign = 1 if action % 2 else -1
            new_departure_time_minutes = self.current['flight_schedule'][flight_name, 0] + sign * self.time_step
            new_arrival_time_minutes = self.current['flight_schedule'][flight_name, 1] + sign * self.time_step
            
            self.current['flight_schedule'][flight_name, 0] = new_departure_time_minutes
            self.current['flight_schedule'][flight_name, 1] = new_arrival_time_minutes
            
            # Update connections
            conn = self.current['connections']
            shift = sign * self.time_step
            try:
                import polars as pl
                if isinstance(conn, pl.DataFrame):
                    delta = (
                        pl.when(pl.col('flight_number_1') == flight_name).then(-shift).otherwise(0)
                        + pl.when(pl.col('flight_number_2') == flight_name).then(shift).otherwise(0)
                    )
                    conn = conn.with_columns((pl.col('cnx_time') + delta).alias('cnx_time'))
                    self.current['connections'] = conn
                else:
                    self.current['connections'][self.current['connections'][:, 0] == flight_name, 4] -= shift
                    self.current['connections'][self.current['connections'][:, 1] == flight_name, 4] += shift
            except Exception:
                try:
                    self.current['connections'][self.current['connections'][:, 0] == flight_name, 4] -= shift
                    self.current['connections'][self.current['connections'][:, 1] == flight_name, 4] += shift
                except Exception:
                    pass
                    
        update_end = time.perf_counter()
        self.timers['update_current'] += update_end - update_start