"""
Tests de performance pour l'optimisation de planning de vol avec RL

Ce module contient des tests de performance pour analyser les temps d'exécution
et les goulots d'étranglement dans l'entraînement et l'évaluation des modèles RL.
"""

import time
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import NormalizeObservation

from schedule_optimizer.flight_scheduling import FlightSchedulingEnv
from schedule_optimizer.utils import generate_random_flight_schedule, generate_lambdas


def test_environment_timers(flight_schedule, lambdas, num_steps=10, max_steps=50):
    """
    Test des timers sur un environnement pour identifier les goulots d'étranglement
    
    Args:
        flight_schedule: DataFrame Polars avec le planning des vols
        lambdas: Dictionnaire des paramètres lambda
        num_steps: Nombre de steps à exécuter pour le test
        max_steps: Nombre max de steps par épisode
    
    Returns:
        dict: Rapport de timing détaillé
    """
    print("🔍 Test des timers sur environnement...")
    
    # Créer un environnement avec timers
    test_env = FlightSchedulingEnv(
        flight_schedule=flight_schedule,
        lambdas=lambdas,
        max_steps=max_steps,
        revenue_estimation='classic', 
        obs_back="numpy"
    )
    
    # Reset timers pour mesures propres
    test_env.reset_timers()
    
    # Faire quelques steps pour collecter des données de timing
    obs, _ = test_env.reset()
    for i in range(num_steps):
        action = test_env.action_space.sample()
        obs, reward, done, _, _ = test_env.step(action)
        if done:
            obs, _ = test_env.reset()
    
    # Récupérer les rapports de timing
    env_timing = test_env.get_timing_report()
    
    revenue_timing = None
    if hasattr(test_env, '_last_revenue_obj'):
        revenue_timing = test_env._last_revenue_obj.get_timing_report()
    
    test_env.close()
    
    return {
        'environment': env_timing,
        'revenue': revenue_timing,
        'num_steps': num_steps
    }


def benchmark_polars_vs_numpy(flight_schedule, lambdas, num_calculations=10, max_steps=50):
    """
    Benchmark des performances Polars vs Numpy pour le calcul de revenue
    
    Args:
        flight_schedule: DataFrame Polars avec le planning des vols
        lambdas: Dictionnaire des paramètres lambda
        num_calculations: Nombre de calculs de revenue à effectuer
        max_steps: Nombre max de steps par épisode
    
    Returns:
        dict: Résultats du benchmark avec temps et speedup
    """
    print("⚡ Benchmark Polars vs Numpy pour calcul de revenue...")
    
    # Test avec Polars (par défaut)
    print("   🔹 Test avec Polars...")
    env_polars = FlightSchedulingEnv(
        flight_schedule=flight_schedule,
        lambdas=lambdas,
        max_steps=max_steps,
        revenue_estimation='classic',
        obs_back="polars"
    )
    
    start_time = time.perf_counter()
    for i in range(num_calculations):
        revenue = env_polars.calculate_revenue()
    end_time = time.perf_counter()
    polars_time = end_time - start_time
    
    env_polars.close()
    
    # Test avec Numpy (conversion forcée)
    print("   🔹 Test avec Numpy...")
    env_numpy = FlightSchedulingEnv(
        flight_schedule=flight_schedule,
        lambdas=lambdas,
        max_steps=max_steps,
        revenue_estimation='classic',
        obs_back="numpy"
    )
    
    start_time = time.perf_counter()
    for i in range(num_calculations):
        revenue = env_numpy.calculate_revenue()
    end_time = time.perf_counter()
    numpy_time = end_time - start_time
    
    env_numpy.close()
    
    # Calculer les métriques
    speedup_ratio = polars_time / numpy_time if numpy_time > 0 else float('inf')
    winner = "Numpy" if numpy_time < polars_time else "Polars"
    
    results = {
        'polars_total_time': polars_time,
        'polars_avg_time_ms': (polars_time / num_calculations) * 1000,
        'numpy_total_time': numpy_time,
        'numpy_avg_time_ms': (numpy_time / num_calculations) * 1000,
        'speedup_ratio': speedup_ratio,
        'winner': winner,
        'num_calculations': num_calculations
    }
    
    print(f"   ✅ Polars: {polars_time:.4f}s total, {results['polars_avg_time_ms']:.2f}ms/calcul")
    print(f"   ✅ Numpy:  {numpy_time:.4f}s total, {results['numpy_avg_time_ms']:.2f}ms/calcul")
    print(f"   🏆 Gagnant: {winner} (speedup: {speedup_ratio:.2f}x)")
    
    return results


def benchmark_ppo_training(flight_schedule, lambdas, total_timesteps=256, max_steps=50):
    """
    Benchmark de l'entraînement PPO avec timing détaillé
    
    Args:
        flight_schedule: DataFrame Polars avec le planning des vols
        lambdas: Dictionnaire des paramètres lambda
        total_timesteps: Nombre total de timesteps d'entraînement
        max_steps: Nombre max de steps par épisode
    
    Returns:
        dict: Métriques de performance de l'entraînement
    """
    print("🚀 Benchmark entraînement PPO avec timing détaillé...")
    
    # Créer un environnement avec timers pour PPO
    ppo_env = FlightSchedulingEnv(
        flight_schedule=flight_schedule,
        lambdas=lambdas,
        max_steps=max_steps,
        revenue_estimation='classic',
        obs_back="numpy"
    )
    ppo_env = NormalizeObservation(ppo_env)
    
    # Reset timers
    ppo_env.env.reset_timers()
    
    # Créer le modèle PPO
    model = PPO("MlpPolicy", ppo_env,
        learning_rate=0.003,
        batch_size=16,
        n_epochs=3,
        clip_range=0.2,
        verbose=0,
        n_steps=128
    )
    
    print(f"   🔹 Début entraînement ({total_timesteps} timesteps)...")
    start_training = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    end_training = time.perf_counter()
    total_training_time = end_training - start_training
    
    # Récupérer les métriques
    env_step_count = ppo_env.env.step_count
    time_per_timestep_ms = (total_training_time / env_step_count) * 1000 if env_step_count > 0 else 0
    
    # Rapports de timing détaillés
    timing_report = ppo_env.env.get_timing_report()
    
    ppo_env.close()
    
    results = {
        'total_training_time': total_training_time,
        'total_timesteps': total_timesteps,
        'env_step_count': env_step_count,
        'time_per_timestep_ms': time_per_timestep_ms,
        'timing_report': timing_report
    }
    
    print(f"   ✅ Entraînement terminé en {total_training_time:.2f}s")
    print(f"   ✅ Total timesteps d'environnement: {env_step_count}")
    print(f"   ✅ Temps par timestep: {time_per_timestep_ms:.2f}ms")
    
    return results


def run_performance_suite(flight_schedule=None, lambdas=None, num_flights=1000):
    """
    Exécuter une suite complète de tests de performance
    
    Args:
        flight_schedule: DataFrame Polars (optionnel, sinon généré aléatoirement)
        lambdas: Dictionnaire des paramètres lambda (optionnel)
        num_flights: Nombre de vols pour le planning généré (si pas fourni)
    
    Returns:
        dict: Résultats complets de tous les tests
    """
    print("🧪 SUITE COMPLÈTE DE TESTS DE PERFORMANCE")
    print("=" * 60)
    
    # Générer un planning si pas fourni
    if flight_schedule is None:
        print(f"🔹 Génération d'un planning aléatoire avec {num_flights} vols...")
        flight_schedule = generate_random_flight_schedule(num_flights)
        lambdas = generate_lambdas(flight_schedule)
    
    results = {}
    
    # 1. Test des timers d'environnement
    print("\n1️⃣ Test des timers d'environnement")
    results['environment_timers'] = test_environment_timers(
        flight_schedule, lambdas, num_steps=10
    )
    
    # 2. Benchmark Polars vs Numpy
    print("\n2️⃣ Benchmark Polars vs Numpy")
    results['polars_vs_numpy'] = benchmark_polars_vs_numpy(
        flight_schedule, lambdas, num_calculations=10
    )
    
    # 3. Benchmark entraînement PPO
    print("\n3️⃣ Benchmark entraînement PPO")
    results['ppo_training'] = benchmark_ppo_training(
        flight_schedule, lambdas, total_timesteps=256
    )
    
    print(f"\n🎉 Suite de tests terminée!")
    return results


def print_performance_summary(results):
    """
    Afficher un résumé des résultats de performance
    
    Args:
        results: Dictionnaire des résultats de run_performance_suite
    """
    print("\n📊 RÉSUMÉ DES PERFORMANCES")
    print("=" * 50)
    
    if 'polars_vs_numpy' in results:
        pn = results['polars_vs_numpy']
        print(f"🔹 Revenue Calculation: {pn['winner']} gagne (speedup: {pn['speedup_ratio']:.2f}x)")
    
    if 'ppo_training' in results:
        ppo = results['ppo_training']
        print(f"🔹 PPO Training: {ppo['time_per_timestep_ms']:.2f}ms par timestep")
    
    if 'environment_timers' in results:
        print(f"🔹 Environment: {results['environment_timers']['num_steps']} steps testés")


if __name__ == "__main__":
    # Exécuter la suite de tests si le script est lancé directement
    print("🚀 Lancement des tests de performance...")
    
    # Générer un planning de test
    test_schedule = generate_random_flight_schedule(100)
    test_lambdas = generate_lambdas(test_schedule)
    
    # Exécuter tous les tests
    results = run_performance_suite(test_schedule, test_lambdas)
    
    # Afficher le résumé
    print_performance_summary(results)