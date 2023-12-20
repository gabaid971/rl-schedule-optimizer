from datetime import datetime, timedelta
import random
import pandas as pd


def generate_random_flight_schedule(N):
    flight_schedule = []

    start_time = datetime.strptime("08:00", "%H:%M")
    end_time = datetime.strptime("22:00", "%H:%M")

    for _ in range(N):
        departure_time = start_time + timedelta(minutes=random.randint(0, (end_time - start_time).seconds // 60))
        arrival_time = departure_time + timedelta(minutes=random.randint(30, 140))  
        airport = random.choice(['MAD', 'JFK', 'FCO'])
        way = random.choice([-1, 1])
        flight_schedule.append((departure_time, arrival_time, way, airport))

    flight_schedule.sort(key=lambda x: x[0])

    random_schedule = pd.DataFrame(flight_schedule, columns=['departure', 'arrival', 'way', 'airport'])

    random_schedule["departure_minutes"] = random_schedule.departure.dt.minute + random_schedule.departure.dt.hour * 60
    random_schedule["arrival_minutes"] = random_schedule.arrival.dt.minute + random_schedule.arrival.dt.hour * 60

    return random_schedule


def transform_schedule(df):
    connections = pd.DataFrame(columns=['flight_number_1', 'flight_number_2', 'leg1', 'leg2', 'cnx_time'])
    for i, departure_flight in df[df['way'] == 1].iterrows():
        arrival_flights = df[(df['way'] == -1) & (df.index != i)]

        for _, arrival_flight in arrival_flights.iterrows():
            departure_j = departure_flight.departure_minutes
            arrival_i = arrival_flight.arrival_minutes
            cnx_time = departure_j - arrival_i
            if arrival_flight.airport != departure_flight.airport:
                new_row = pd.DataFrame({
                    'flight_number_1' : arrival_flight.name,
                    'flight_number_2' : departure_flight.name,
                    'leg1' : arrival_flight.airport, 
                    'leg2' : departure_flight.airport,
                    'cnx_time' : cnx_time
                }, index = [0])
                connections = pd.concat([new_row, connections[:]], axis= 0)
    return connections


def generate_lambdas(df):
    lst = df.airport.unique().tolist()
    lambdas = {}
    
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j:
                ond = ''.join(lst[i] + lst[j])
                dno = ''.join(lst[j] + lst[i])
                if ond not in lambdas:
                    #lambda_ = random.randint(0, 100)
                    lambda_ = 1000
                    lambdas[ond] = lambda_
                    lambdas[dno] = lambda_
    return lambdas