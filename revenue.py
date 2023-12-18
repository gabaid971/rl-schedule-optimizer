from datetime import timedelta
import pandas as pd


class Revenue:
    def __init__(self, current, lambdas, min_connection_time=60, max_connection_time=480):
        self.current = current
        self.lambdas = lambdas
        self.min_connection_time = min_connection_time
        self.max_connection_time = max_connection_time

    def calculate_feasible_connections_old(self):
        connections = self.current['connections']
        connections['feasible'] = connections['cnx_time'].\
            between(self.min_connection_time, self.max_connection_time)
        connections['Lambda'] = connections.apply(
            lambda x: self.lambdas[x.leg1 + x.leg2],
            axis = 1
        )
        connections['preference_value'] = connections['cnx_time'].\
            apply(self.preference_curve)
        connections['revenue'] = connections['Lambda'] * connections['preference_value']
        return connections 
    
    def calculate_revenue(self):
        total_revenue = 0
        connections = self.current['connections']
        for connection in connections:
            cnx_time = connection[4]
            if cnx_time >= self.min_connection_time\
                and cnx_time <= self.max_connection_time:
                leg1 = connection[2]
                leg2 = connection[3]
                Lambda = self.lambdas[leg1 + leg2]
                preference_value = self.preference_curve(cnx_time)
                cnx_revenue = preference_value * Lambda
                total_revenue += cnx_revenue
        return total_revenue

    def preference_curve(self, x):
        if 60 <= x <= 120:
            return 0.5 + 0.5 * (x - 60) / 60
        elif 120 < x < 480:
            return 1 - 0.5 * (x - 120) / 360
        else:
            return 0
        
    def calculate_revenue_old(self):
        connections = self.calculate_feasible_connections()
        if len(connections) > 0:
            return connections['revenue'].sum()
        else:
            return 0