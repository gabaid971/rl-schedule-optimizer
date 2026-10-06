import numpy as np


class RewardTracker:
    def __init__(self, name):
        self.name = name
        self.episode_rewards = []
        self.steps = []
    
    def record_episode(self, step, reward):
        """Enregistre un reward à un step donné"""
        self.steps.append(step)
        self.episode_rewards.append(reward)
    
    def get_data(self):
        """Retourne les données pour plotting"""
        return self.steps, self.episode_rewards
    
    def get_final_stats(self):
        """Statistiques finales"""
        if not self.episode_rewards:
            return {"final": 0, "max": 0, "mean": 0}
        
        return {
            "final": self.episode_rewards[-1],
            "max": max(self.episode_rewards),
            "mean": np.mean(self.episode_rewards)
        }

def evaluate_model(model, env, masked=False, num_episodes=5):
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if masked:
                action_mask = env.env.env.get_action_mask()
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            else:
                action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def run_evaluation(TOTAL_TIMESTEPS, EVAL_FREQ, model, trackers, env, masked=False):
    for step in range(0, TOTAL_TIMESTEPS, EVAL_FREQ):
        model.learn(total_timesteps=EVAL_FREQ)

        reward = evaluate_model(model, env, masked=masked)
        # trackers['DQN'].record_episode(step + EVAL_FREQ, reward)
        
        if step % (EVAL_FREQ * 4) == 0:
            print(f"  Step {step + EVAL_FREQ:4d}: Reward = {reward:6.2f}")
    return model