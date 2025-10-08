from datetime import datetime, timedelta
import random
import polars as pl


def generate_random_flight_schedule(N):
    flight_schedule_dep = []
    flight_schedule_arr = []
    flight_schedule_way = []
    flight_schedule_airport = []

    start_time = datetime.strptime("08:00", "%H:%M")
    end_time = datetime.strptime("22:00", "%H:%M")

    for _ in range(N):
        departure_time = start_time + timedelta(minutes=random.randint(0, (end_time - start_time).seconds // 60))
        arrival_time = departure_time + timedelta(minutes=random.randint(30, 140))  
        airport = random.choice(['MAD', 'JFK', 'FCO'])
        way = random.choice([-1, 1])
        
        flight_schedule_dep.append(departure_time)
        flight_schedule_arr.append(arrival_time)
        flight_schedule_way.append(way)
        flight_schedule_airport.append(airport)

    # Trier par heure de départ
    sorted_data = sorted(zip(flight_schedule_dep, flight_schedule_arr, flight_schedule_way, flight_schedule_airport), key=lambda x: x[0])
    flight_schedule_dep, flight_schedule_arr, flight_schedule_way, flight_schedule_airport = zip(*sorted_data)

    random_schedule = pl.DataFrame({
        'departure': flight_schedule_dep,
        'arrival': flight_schedule_arr, 
        'way': flight_schedule_way,
        'airport': flight_schedule_airport
    })

    random_schedule = random_schedule.with_columns([
        (pl.col('departure').dt.hour().cast(pl.Int32) * 60 + pl.col('departure').dt.minute().cast(pl.Int32)).alias('departure_minutes'),
        (pl.col('arrival').dt.hour().cast(pl.Int32) * 60 + pl.col('arrival').dt.minute().cast(pl.Int32)).alias('arrival_minutes')
    ])

    return random_schedule


def transform_schedule_old(df):
    connections_data = []
    
    # Récupérer les données du DataFrame avec les index absolus
    df_with_index = df.with_row_index("row_index")
    
    # Filtrer les vols de départ (way == 1) et d'arrivée (way == -1)
    departure_flights = df_with_index.filter(pl.col('way') == 1)
    arrival_flights = df_with_index.filter(pl.col('way') == -1)
    
    for departure_row in departure_flights.iter_rows(named=True):
        for arrival_row in arrival_flights.iter_rows(named=True):
            departure_j = departure_row['departure_minutes']
            arrival_i = arrival_row['arrival_minutes']
            cnx_time = departure_j - arrival_i
            
            if arrival_row['airport'] != departure_row['airport']:
                connections_data.append({
                    'flight_number_1': arrival_row['row_index'],   # Index absolut de l'arrival flight
                    'flight_number_2': departure_row['row_index'], # Index absolut du departure flight  
                    'leg1': arrival_row['airport'], 
                    'leg2': departure_row['airport'],
                    'cnx_time': cnx_time
                })
    
    connections = pl.DataFrame(connections_data)
    # Convertir en numpy array pour compatibilité avec le reste du code
    return connections.to_numpy()

import polars as pl

def transform_schedule(df: pl.DataFrame) -> pl.DataFrame:
    # Séparer départs et arrivées avec un index absolu
    df_with_index = df.with_row_index("row_index")
    departures = df_with_index.filter(pl.col("way") == 1).rename(
        {"row_index": "flight_number_2", "airport": "leg2", "departure": "dep_time"}
    )
    arrivals = df_with_index.filter(pl.col("way") == -1).rename(
        {"row_index": "flight_number_1", "airport": "leg1", "arrival": "arr_time"}
    )

    # Produit cartésien entre arrivals et departures
    connections = arrivals.join(departures, how="cross")

    # Calcul du temps de connexion en minutes
    connections = connections.with_columns(
        ((pl.col("dep_time") - pl.col("arr_time")).dt.total_seconds() / 60).alias("cnx_time")
    )

    # On filtre pour éviter leg1 == leg2
    connections = connections.filter(pl.col("leg1") != pl.col("leg2"))

    # On garde uniquement les colonnes utiles
    return connections.select(
        ["flight_number_1", "flight_number_2", "leg1", "leg2", "cnx_time"]
    )



def generate_lambdas(df):
    lst = df['airport'].unique().to_list()
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