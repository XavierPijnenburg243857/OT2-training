from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from ot2_env_wrapper import OT2Env
from clearml import Task

Task.add_requirements('requirements.txt')

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--early_stopping_patience", type=int, default=5)
parser.add_argument("--n_eval_episodes", type=int, default=50)
parser.add_argument("--success_threshold", type=float, default=0.001)
parser.add_argument("--min_success_improvement", type=float, default=0.05)

args = parser.parse_args()

task = Task.init(
    project_name='Mentor Group - Uther/Group 1',
    task_name=f'OT2_LR{args.learning_rate}_BS{args.batch_size}'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

os.environ['WANDB_API_KEY'] = 'b82d3dc4d93780790f8600c249009ba32eb79bd4'

run = wandb.init(project="ot2_robot_training", sync_tensorboard=True)

env = OT2Env(render=False, max_steps=args.max_steps)


def evaluate_with_success_rate(model, env, n_eval_episodes=50, success_threshold=0.001):
    """
    Evaluate model and return detailed metrics including success rate.
    
    Args:
        model: Trained PPO model
        env: Environment
        n_eval_episodes: Number of episodes to evaluate
        success_threshold: Distance threshold for success (meters)
    
    Returns:
        dict with success_rate, avg_steps, avg_final_distance
    """
    successes = 0
    successful_steps = []
    all_distances = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        final_dist = info['distance']
        all_distances.append(final_dist)
        
        if final_dist < success_threshold:
            successes += 1
            successful_steps.append(steps)
    
    return {
        'success_rate': successes / n_eval_episodes,
        'avg_steps': sum(successful_steps) / len(successful_steps) if successful_steps else float('inf'),
        'avg_final_distance': sum(all_distances) / len(all_distances),
        'num_successes': successes,
        'num_episodes': n_eval_episodes
    }


model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=f"runs/{run.id}")

wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

time_steps = 100000
best_success_rate = -1.0
best_iteration = 0
no_improvement_count = 0

for i in range(20):
    model.learn(
        total_timesteps=time_steps, 
        callback=wandb_callback, 
        progress_bar=True, 
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    
    # Evaluate with detailed metrics
    metrics = evaluate_with_success_rate(
        model, 
        env, 
        n_eval_episodes=args.n_eval_episodes,
        success_threshold=args.success_threshold
    )
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"Cycle {i+1}/20 Evaluation Results:")
    print(f"{'='*60}")
    print(f"  Success Rate:      {metrics['success_rate']*100:.1f}% ({metrics['num_successes']}/{metrics['num_episodes']})")
    print(f"  Avg Steps (success): {metrics['avg_steps']:.1f}")
    print(f"  Avg Final Distance: {metrics['avg_final_distance']*1000:.2f}mm")
    print(f"{'='*60}\n")
    
    # Log to wandb
    wandb.log({
        "eval/success_rate": metrics['success_rate'],
        "eval/avg_steps": metrics['avg_steps'],
        "eval/avg_final_distance_mm": metrics['avg_final_distance'] * 1000,
        "cycle": i+1
    })
    
    # Save model after each cycle
    model.save(f"models/{run.id}/{time_steps*(i+1)}")

    # Always track and save best model
    if metrics['success_rate'] > best_success_rate:
        best_success_rate = metrics['success_rate']
        best_iteration = i + 1
        model.save(f"models/{run.id}/best_model")
        print(f"New best: {best_success_rate*100:.1f}%")

    # Early stopping based on meaningful improvement
    if i > 0 and metrics['success_rate'] > best_success_rate - args.min_success_improvement:
        no_improvement_count = 0  # Reset if within threshold of best
    else:
        no_improvement_count += 1
        print(f"No improvement for {no_improvement_count} cycle(s)")
        print(f"   (Best: {best_success_rate*100:.1f}% from cycle {best_iteration})")
    
    if no_improvement_count >= args.early_stopping_patience:
        print(f"\n{'='*60}")
        print(f"Early stopping triggered!")
        print(f"   No improvement for {args.early_stopping_patience} consecutive cycles")
        print(f"   Best success rate: {best_success_rate*100:.1f}% (Cycle {best_iteration})")
        print(f"{'='*60}\n")
        print(f"Loading best model from cycle {best_iteration}...")
        model = PPO.load(f"models/{run.id}/best_model", env=env)
        break

# Final evaluation with more episodes
print(f"\n{'='*60}")
print(f"Final Model Evaluation (250 episodes)")
print(f"{'='*60}")
final_metrics = evaluate_with_success_rate(
    model, 
    env, 
    n_eval_episodes=250,
    success_threshold=args.success_threshold
)
print(f"  Final Success Rate:     {final_metrics['success_rate']*100:.1f}%")
print(f"  Final Avg Steps:        {final_metrics['avg_steps']:.1f}")
print(f"  Final Avg Distance:     {final_metrics['avg_final_distance']*1000:.2f}mm")
print(f"{'='*60}\n")

wandb.log({
    "final/success_rate": final_metrics['success_rate'],
    "final/avg_steps": final_metrics['avg_steps'],
    "final/avg_final_distance_mm": final_metrics['avg_final_distance'] * 1000
})

env.close()
wandb.finish()

model.save("model_name")
task.upload_artifact("model", artifact_object="model_name.zip")