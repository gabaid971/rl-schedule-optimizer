"""
Optimisation itérative de planning de vol avec RL et multi-sampling

Ce module contient la classe IterativeOptimizer pour l'optimisation progressive
de plannings de vol en utilisant des modèles de reinforcement learning avec
action masking et extraction multi-sampling.
"""

import time
import polars as pl
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.wrappers import NormalizeObservation

from schedule_optimizer.flight_scheduling_optimized import FlightSchedulingEnvMasked
from schedule_optimizer.model_evaluation import evaluate_model


def extract_optimized_schedule(
    model, 
    env_base, 
    original_schedule: pl.DataFrame, 
    max_steps: int = 50, 
    n_samples: int = 10
) -> Tuple[pl.DataFrame, float, float]:
    """
    Extraire le planning optimisé en générant n chemins et gardant le meilleur
    
    Args:
        model: Modèle MaskablePPO entraîné
        env_base: Environnement avec wrappers (ActionMasker + NormalizeObservation)
        original_schedule: Planning original (DataFrame Polars)
        max_steps: Nombre max de steps pour l'optimisation
        n_samples: Nombre de chemins à tester (par défaut 10)
    
    Returns:
        optimized_schedule: Meilleur planning optimisé (DataFrame Polars)
        best_total_reward: Meilleur reward total obtenu
        best_final_revenue: Meilleur revenue final
    """
    print(f"🔍 Extraction du planning optimisé avec {n_samples} échantillons...")
    
    best_total_reward = float('-inf')
    best_final_revenue = 0
    best_schedule = None
    best_step_count = 0
    
    for sample_idx in range(n_samples):
        # Reset de l'environnement pour chaque échantillon
        obs, _ = env_base.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        # Exécution du modèle (déterministe ou stochastique selon l'échantillon)
        deterministic = (sample_idx == 0)  # Premier échantillon déterministe, autres stochastiques
        
        while not done:
            # Obtenir le masque d'actions valides
            action_mask = env_base.env.env.get_action_mask()
            
            # Prédiction du modèle
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=deterministic)
            
            # Exécuter l'action
            obs, reward, done, _, _ = env_base.step(action)
            total_reward += reward
            step_count += 1
        
        # Récupérer les métriques finales pour cet échantillon
        final_flight_schedule = env_base.env.env.current['flight_schedule']
        final_revenue = env_base.env.env.calculate_revenue()
        
        # Garder le meilleur échantillon (basé sur le reward total)
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_final_revenue = final_revenue
            best_step_count = step_count
            
            # Sauvegarder le meilleur planning
            best_schedule = original_schedule.clone()
            
            # Mettre à jour avec les nouveaux horaires (en minutes)
            new_departure_minutes = final_flight_schedule[:, 0].tolist()
            new_arrival_minutes = final_flight_schedule[:, 1].tolist()
            
            best_schedule = best_schedule.with_columns([
                pl.Series("departure_minutes", new_departure_minutes),
                pl.Series("arrival_minutes", new_arrival_minutes)
            ])
            
            # Recalculer les datetime
            base_date = datetime(1900, 1, 1, 0, 0, 0)
            
            best_schedule = best_schedule.with_columns([
                pl.lit(base_date).dt.offset_by(pl.col('departure_minutes').cast(pl.String) + "m").alias('departure'),
                pl.lit(base_date).dt.offset_by(pl.col('arrival_minutes').cast(pl.String) + "m").alias('arrival')
            ])
    
    print(f"   ✅ MEILLEUR échantillon: reward={best_total_reward:.2f}, revenue={best_final_revenue:.2f}, steps={best_step_count}")
    print(f"   📈 Amélioration sur {n_samples} essais")
    
    return best_schedule, best_total_reward, best_final_revenue


class IterativeOptimizer:
    """
    Optimisateur itératif qui améliore progressivement le planning
    
    Cette classe implémente une approche d'optimisation itérative où :
    1. Un modèle RL est entraîné sur le planning actuel
    2. Le meilleur planning est extrait via multi-sampling
    3. Si le nouveau planning est meilleur, on l'adopte pour la prochaine itération
    4. Le processus continue jusqu'à convergence ou nombre max d'itérations
    """
    
    def __init__(
        self, 
        base_schedule: pl.DataFrame, 
        lambdas: Dict, 
        max_steps: int = 50, 
        n_samples: int = 10
    ):
        """
        Initialiser l'optimisateur itératif
        
        Args:
            base_schedule: Planning initial (DataFrame Polars)
            lambdas: Dictionnaire des paramètres lambda pour l'environnement
            max_steps: Nombre max de steps par épisode
            n_samples: Nombre d'échantillons pour l'extraction multi-sampling
        """
        self.base_schedule = base_schedule
        self.lambdas = lambdas
        self.max_steps = max_steps
        self.n_samples = n_samples
        self.iteration_history = []
        self.best_schedule = base_schedule.clone()
        self.best_revenue = None
        
    def run_iteration(
        self, 
        current_schedule: pl.DataFrame, 
        iteration_num: int, 
        timesteps_per_iteration: int = 10000
    ) -> Tuple[pl.DataFrame, Dict]:
        """
        Exécuter une itération d'optimisation complète
        
        Args:
            current_schedule: Planning actuel pour cette itération
            iteration_num: Numéro de l'itération (pour logging)
            timesteps_per_iteration: Nombre de timesteps d'entraînement
        
        Returns:
            next_schedule: Planning à utiliser pour la prochaine itération
            iteration_result: Dictionnaire avec les métriques de l'itération
        """
        print(f"🔄 ITÉRATION {iteration_num}")
        print(f"   Timesteps par itération: {timesteps_per_iteration}")
        print(f"   Échantillons d'extraction: {self.n_samples}")
        
        # 1. Créer l'environnement avec le planning actuel
        env_base = FlightSchedulingEnvMasked(
            current_schedule,
            self.lambdas,
            max_steps=self.max_steps,
            revenue_estimation='classic',
            obs_back="numpy"
        )
        
        def mask_fn(env):
            return env.get_action_mask()
        
        env = ActionMasker(env_base, mask_fn)
        env = NormalizeObservation(env)
        
        # 2. Créer et entraîner un nouveau modèle
        model = MaskablePPO("MlpPolicy", env,
            learning_rate=0.003,
            batch_size=128,
            n_epochs=10,
            clip_range=0.2,
            verbose=0
        )
        
        print(f"   🚀 Entraînement en cours...")
        start_time = time.time()
        model.learn(total_timesteps=timesteps_per_iteration)
        
        # Évaluation rapide post-entraînement
        eval_reward = evaluate_model(model, env, masked=True, num_episodes=5)
        print(f"   📊 Évaluation post-entraînement: {eval_reward:.2f}")
        
        training_time = time.time() - start_time
        
        # 3. Extraire le planning optimisé avec MULTI-SAMPLING
        optimized_schedule, total_reward, final_revenue = extract_optimized_schedule(
            model, env, current_schedule, self.max_steps, n_samples=self.n_samples
        )
        
        # 4. Calculer les métriques d'amélioration
        env_current = FlightSchedulingEnvMasked(
            current_schedule, self.lambdas, max_steps=self.max_steps, 
            revenue_estimation='classic', obs_back="numpy"
        )
        current_revenue = env_current.calculate_revenue()
        improvement = final_revenue - current_revenue
        
        # 5. DÉCISION : Garder le planning optimisé seulement si amélioration
        if improvement > 0:
            # ✅ AMÉLIORATION : Accepter le nouveau planning
            next_schedule = optimized_schedule.clone()
            self.best_schedule = optimized_schedule.clone()
            self.best_revenue = final_revenue
            status = "✅ ACCEPTÉ"
            print(f"   {status} Revenue: {current_revenue:.2f} → {final_revenue:.2f} (+{improvement:.2f}, +{improvement/current_revenue*100:.1f}%)")
        else:
            # ❌ PAS D'AMÉLIORATION : Garder le planning actuel
            next_schedule = current_schedule.clone()
            status = "❌ REJETÉ"
            print(f"   {status} Revenue: {current_revenue:.2f} → {final_revenue:.2f} ({improvement:.2f}, {improvement/current_revenue*100:.1f}%)")
            print(f"   🔄 Maintien du planning actuel")
        
        # Initialiser best_revenue à la première itération
        if self.best_revenue is None:
            self.best_revenue = current_revenue
            self.best_schedule = current_schedule.clone()
        
        # 6. Stocker les résultats
        iteration_result = {
            'iteration': iteration_num,
            'timesteps': timesteps_per_iteration,
            'n_samples': self.n_samples,
            'training_time': training_time,
            'eval_reward': eval_reward,
            'current_revenue': current_revenue,
            'optimized_revenue': final_revenue,
            'improvement': improvement,
            'improvement_pct': (improvement / current_revenue * 100) if current_revenue != 0 else 0,
            'total_reward': total_reward,
            'schedule': optimized_schedule,
            'accepted': improvement > 0,
            'next_schedule': next_schedule,
            'status': status,
            'best_revenue_so_far': self.best_revenue
        }
        
        self.iteration_history.append(iteration_result)
        print(f"   ⏱️  Temps d'entraînement: {training_time:.1f}s")
        print(f"   🏆 Meilleur revenue global: {self.best_revenue:.2f}")
        
        # 7. Nettoyer
        env.close()
        env_current.close()
        
        return next_schedule, iteration_result
    
    def optimize_iteratively(
        self, 
        num_iterations: int = 5, 
        timesteps_per_iteration: int = 8000,
        max_consecutive_rejections: int = 3
    ) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Exécuter plusieurs itérations d'optimisation
        
        Args:
            num_iterations: Nombre maximum d'itérations
            timesteps_per_iteration: Timesteps d'entraînement par itération
            max_consecutive_rejections: Arrêt si trop de rejets consécutifs
        
        Returns:
            best_schedule: Meilleur planning trouvé
            iteration_history: Historique détaillé de toutes les itérations
        """
        print(f"🎯 OPTIMISATION ITÉRATIVE AVEC MULTI-SAMPLING - {num_iterations} itérations")
        print(f"🔬 Paramètres: {self.n_samples} échantillons par extraction")
        print("="*70)
        
        current_schedule = self.base_schedule.clone()
        consecutive_rejections = 0
        
        for i in range(1, num_iterations + 1):
            current_schedule, result = self.run_iteration(
                current_schedule, i, timesteps_per_iteration
            )
            
            # Gestion des rejets consécutifs
            if not result['accepted']:
                consecutive_rejections += 1
            else:
                consecutive_rejections = 0
            
            # Arrêter si trop de rejets consécutifs (convergence probable)
            if consecutive_rejections >= max_consecutive_rejections:
                print(f"⚠️  Convergence détectée : {consecutive_rejections} rejets consécutifs.")
                print(f"   Arrêt anticipé à l'itération {i}.")
                break
                
        return self.best_schedule, self.iteration_history
    
    def print_summary(self):
        """Afficher le résumé des itérations avec statistiques d'acceptation"""
        print(f"📊 RÉSUMÉ OPTIMISATION ITÉRATIVE AVEC MULTI-SAMPLING")
        print(f"🔬 Configuration: {self.n_samples} échantillons par extraction")
        print("="*70)
        
        if not self.iteration_history:
            print("Aucune itération exécutée.")
            return
            
        total_time = sum(iter_result['training_time'] for iter_result in self.iteration_history)
        total_timesteps = sum(iter_result['timesteps'] for iter_result in self.iteration_history)
        
        # Statistiques d'acceptation
        accepted_count = sum(1 for result in self.iteration_history if result['accepted'])
        rejected_count = len(self.iteration_history) - accepted_count
        acceptance_rate = (accepted_count / len(self.iteration_history)) * 100
        
        print(f"Nombre d'itérations: {len(self.iteration_history)}")
        print(f"Itérations acceptées: {accepted_count} ({acceptance_rate:.1f}%)")
        print(f"Itérations rejetées: {rejected_count} ({100-acceptance_rate:.1f}%)")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Temps total: {total_time:.1f}s")
        
        print(f"📈 ÉVOLUTION DU REVENUE:")
        for iter_result in self.iteration_history:
            status_emoji = "✅" if iter_result['accepted'] else "❌"
            eval_info = f"eval={iter_result['eval_reward']:5.1f}" if 'eval_reward' in iter_result else ""
            print(f"  Itération {iter_result['iteration']:2d}: "
                  f"{iter_result['current_revenue']:7.2f} → {iter_result['optimized_revenue']:7.2f} "
                  f"({iter_result['improvement']:+5.2f}, {iter_result['improvement_pct']:+5.1f}%) {status_emoji} {eval_info}")
        
        if len(self.iteration_history) > 0:
            first_revenue = self.iteration_history[0]['current_revenue']
            final_best_revenue = self.best_revenue
            total_improvement = final_best_revenue - first_revenue
            
            print(f"🎯 AMÉLIORATION TOTALE:")
            print(f"   Initial: {first_revenue:.2f}")
            print(f"   Meilleur: {final_best_revenue:.2f}")
            print(f"   Gain:    +{total_improvement:.2f} ({(total_improvement/first_revenue*100):+.1f}%)")
            print(f"   ROI:     {total_improvement/total_time:.3f} revenue/seconde")
            
            # Efficacité des acceptations
            if accepted_count > 0:
                avg_improvement_per_accepted = total_improvement / accepted_count
                print(f"   Efficacité: {avg_improvement_per_accepted:.2f} revenue/itération acceptée")
    
    def save_best_schedule(self, filepath: str):
        """
        Sauvegarder le meilleur planning trouvé
        
        Args:
            filepath: Chemin du fichier de sauvegarde (format parquet recommandé)
        """
        if self.best_schedule is not None:
            self.best_schedule.write_parquet(filepath)
            print(f"💾 Meilleur planning sauvegardé: {filepath}")
        else:
            print("❌ Aucun planning à sauvegarder")
    
    def get_optimization_metrics(self) -> Dict:
        """
        Obtenir les métriques d'optimisation sous forme de dictionnaire
        
        Returns:
            dict: Métriques consolidées de l'optimisation
        """
        if not self.iteration_history:
            return {}
        
        accepted_count = sum(1 for result in self.iteration_history if result['accepted'])
        total_time = sum(iter_result['training_time'] for iter_result in self.iteration_history)
        first_revenue = self.iteration_history[0]['current_revenue']
        total_improvement = self.best_revenue - first_revenue if self.best_revenue else 0
        
        return {
            'num_iterations': len(self.iteration_history),
            'accepted_iterations': accepted_count,
            'acceptance_rate': (accepted_count / len(self.iteration_history)) * 100,
            'total_training_time': total_time,
            'initial_revenue': first_revenue,
            'best_revenue': self.best_revenue,
            'total_improvement': total_improvement,
            'improvement_percentage': (total_improvement / first_revenue * 100) if first_revenue != 0 else 0,
            'roi_revenue_per_second': total_improvement / total_time if total_time > 0 else 0
        }


def optimize_schedule(
    flight_schedule: pl.DataFrame,
    lambdas: Dict,
    num_iterations: int = 5,
    timesteps_per_iteration: int = 10000,
    n_samples: int = 10,
    max_steps: int = 50
) -> Tuple[pl.DataFrame, Dict]:
    """
    Fonction utilitaire pour optimiser un planning de vol
    
    Args:
        flight_schedule: Planning initial
        lambdas: Paramètres lambda pour l'environnement
        num_iterations: Nombre d'itérations d'optimisation
        timesteps_per_iteration: Timesteps d'entraînement par itération
        n_samples: Nombre d'échantillons pour l'extraction
        max_steps: Steps max par épisode
    
    Returns:
        optimized_schedule: Meilleur planning optimisé
        metrics: Métriques d'optimisation
    """
    optimizer = IterativeOptimizer(
        flight_schedule, lambdas, max_steps=max_steps, n_samples=n_samples
    )
    
    best_schedule, _ = optimizer.optimize_iteratively(
        num_iterations=num_iterations,
        timesteps_per_iteration=timesteps_per_iteration
    )
    
    optimizer.print_summary()
    metrics = optimizer.get_optimization_metrics()
    
    return best_schedule, metrics


if __name__ == "__main__":
    # Exemple d'utilisation si le script est lancé directement
    from schedule_optimizer.utils import generate_random_flight_schedule, generate_lambdas
    
    print("🚀 Test d'optimisation itérative...")
    
    # Générer un planning de test
    test_schedule = generate_random_flight_schedule(20)
    test_lambdas = generate_lambdas(test_schedule)
    
    # Optimiser avec des paramètres de test (rapides)
    optimized_schedule, metrics = optimize_schedule(
        test_schedule,
        test_lambdas,
        num_iterations=2,
        timesteps_per_iteration=1000,
        n_samples=5
    )
    
    print(f"🎉 Optimisation terminée!")
    print(f"📊 Amélioration: {metrics.get('improvement_percentage', 0):.1f}%")
    print(f"💰 Revenue final: {metrics.get('best_revenue', 0):.2f}")