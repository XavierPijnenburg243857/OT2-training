from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from ot2_env_wrapper import OT2Env
from clearml import Task

Task.add_requirements('requirements.txt')

# MOVE ARGPARSE HERE - BEFORE Task.init()
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
parser.add_argument("--early_stopping_patience", type=int, default=3)
args = parser.parse_args()

# NOW initialize ClearML - it will capture the args
task = Task.init(
    project_name='Mentor Group - Uther/Group 1',
    task_name=f'OT2_LR{args.learning_rate}_BS{args.batch_size}'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Rest of your script...
os.environ['WANDB_API_KEY'] = 'b82d3dc4d93780790f8600c249009ba32eb79bd4'

run = wandb.init(project="ot2_robot_training", sync_tensorboard=True)

env = OT2Env(render=False, max_steps=args.max_steps)

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
best_reward = -float('inf')
best_iteration = 0
no_improvement_count = 0

for i in range(10):
    model.learn(
        total_timesteps=time_steps, 
        callback=wandb_callback, 
        progress_bar=True, 
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\n✅ Cycle {i+1} - Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({"eval/mean_reward": mean_reward, "cycle": i+1})
    model.save(f"models/{run.id}/{time_steps*(i+1)}")
    
    if mean_reward > best_reward + 5:
        best_reward = mean_reward
        best_iteration = i + 1
        no_improvement_count = 0
        model.save(f"models/{run.id}/best_model")
        print(f"✅ New best reward: {best_reward:.2f}")
    else:
        no_improvement_count += 1
        print(f"⚠️ No improvement for {no_improvement_count} cycles")
    
    if no_improvement_count >= args.early_stopping_patience:
        print(f"🛑 Early stopping: No improvement for {args.early_stopping_patience} cycles")
        print(f"📊 Loading best model from iteration {best_iteration}")
        model = PPO.load(f"models/{run.id}/best_model", env=env)
        break

env.close()
wandb.finish()

model.save("model_name")
task.upload_artifact("model", artifact_object="model_name.zip")