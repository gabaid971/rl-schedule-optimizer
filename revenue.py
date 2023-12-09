from datetime import timedelta
import pandas as pd


class Revenue:
    def __init__(self, current_state, lambdas, min_connection_time=60, max_connection_time=480):
        self.current_state = current_state
        self.lambdas = lambdas
        self.min_connection_time = min_connection_time
        self.max_connection_time = max_connection_time

    def calculate_feasible_connections(self):
        flights = self.current_state
        connections = []
        for i in range(len(flights)):
            for j in range(len(flights)):
                if i != j and flights[i, 2] == 0 and flights[j, 2] == 1:  
                    departure_j = flights[j, 0]
                    arrival_i = flights[i, 1]
                    cnx_time = departure_j - arrival_i
                    if self.min_connection_time <= cnx_time <= self.max_connection_time:
                        preference_value = self.preference_curve(cnx_time)
                        lambda_value = self.lambdas[(i, j)]
                        revenue = preference_value * lambda_value
                        connections.append({
                            'leg_1': i,
                            'leg_2': j,
                            'cnx_time': cnx_time,
                            'preference_value' : preference_value,
                            'lambda_value' : lambda_value,
                            'revenue' : revenue
                        })
        return connections

    def preference_curve(self, x):
        if 60 <= x <= 120:
            return 0.5 + 0.5 * (x - 60) / 60
        elif 120 < x < 480:
            return 1 - 0.5 * (x - 120) / 360
        else:
            return 0
        
    def calculate_revenue(self):
        connections = self.calculate_feasible_connections()
        if len(connections) > 0:
            return sum(connection['revenue'] for connection in connections)
        else:
            return 0
        
    def calculate_feasible_connections_old(self):
        df = self.current_state
        departures = df[df['way'] == 1].copy()
        arrivals = df[df['way'] == -1].copy()

        feasible_connections = pd.DataFrame(columns=['departure_flight', 'arrival_flight', 'cnx_time'])
        for i in range(len(arrivals)):
            arrival_flight = arrivals.iloc[i]
            
            possible_departures = departures[
                (departures['departure'] >= arrival_flight['arrival'] +\
                  timedelta(minutes=self.min_connection_time)) &
                (departures['departure'] <= arrival_flight['arrival'] +\
                  timedelta(minutes=self.max_connection_time))
            ]
            for _, departure_flight in possible_departures.iterrows():
                connection_time = departure_flight['departure'] - arrival_flight['arrival']
                connection_info = {
                    'departure_flight': departure_flight.name,
                    'arrival_flight': arrival_flight.name,
                    'cnx_time': connection_time,
                    'cnx_time_min' : int(connection_time.total_seconds() / 60)
                }
                feasible_connections = pd.concat([feasible_connections, 
                    pd.DataFrame([connection_info])], 
                    ignore_index=True
                )
        return feasible_connections
    
    def calculate_revenue_old(self):
        connections = self.calculate_feasible_connections()
        if len(connections) > 0:
            connections['z'] = connections['cnx_time_min'].apply(lambda x : self.preference_curve(x))
            connections['lambdas'] = connections.apply(lambda x : 
                self.lambdas[(x.departure_flight, x.arrival_flight)], 
                axis=1
            )
            connections['revenue'] = connections['z'] * connections['lambdas']
            return connections.revenue.sum()
        else:
            return 0
        
