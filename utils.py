from datetime import datetime, timedelta
import random
import pandas as pd


def generate_random_flight_schedule(N):
    flight_schedule = []

    start_time = datetime.strptime("08:00", "%H:%M")
    end_time = datetime.strptime("22:00", "%H:%M")

    for _ in range(N):
        departure_time = start_time + timedelta(minutes=random.randint(0, (end_time - start_time).seconds // 60))
        arrival_time = departure_time + timedelta(minutes=random.randint(30, 140))  # Exemple : vol de 30 à 140 minutes

        flight_schedule.append((departure_time, arrival_time, random.choice([-1, 1])))

    flight_schedule.sort(key=lambda x: x[0])

    random_schedule = pd.DataFrame(flight_schedule, columns=['departure', 'arrival', 'way'])

    random_schedule["departure_minutes"] = random_schedule.departure.dt.minute + random_schedule.departure.dt.hour * 60
    random_schedule["arrival_minutes"] = random_schedule.arrival.dt.minute + random_schedule.arrival.dt.hour * 60
    random_schedule["way_transformed"] = random_schedule["way"].replace(-1, 0)

    return random_schedule


def generate_lambdas(df):
    lambdas = {}
    for i, departure_flight in df[df['way'] == 1].iterrows():
        arrival_flights = df[(df['way'] == -1) & (df.index != i)]

        for _, arrival_flight in arrival_flights.iterrows():
            lambdas[(arrival_flight.name, departure_flight.name)] = random.randint(0, 100)
    return lambdas