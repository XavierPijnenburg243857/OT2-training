from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from ot2_env_wrapper import OT2Env  # Import your custom environment
from clearml import Task

task.add_requirements('requirements.txt')

task = Task.init(project_name='Mentor Group - Uther/Group 1', # NB: Replace YourName with your own name
                    task_name='OT2-experiment-1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)

args = parser.parse_args()

os.environ['WANDB_API_KEY'] = 'b82d3dc4d93780790f8600c249009ba32eb79bd4'

# Initialize wandb project
run = wandb.init(project="ot2_robot_training", sync_tensorboard=True)

# Create your custom environment instead of Pendulum
env = OT2Env(render=False, max_steps=args.max_steps)

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}")

# Create wandb callback
wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

time_steps = 100000
for i in range(10):
    model.learn(
        total_timesteps=time_steps, 
        callback=wandb_callback, 
        progress_bar=True, 
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    
    # Evaluate after each cycle
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\n✅ Cycle {i+1} - Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    wandb.log({"eval/mean_reward": mean_reward, "cycle": i+1})
    model.save(f"models/{run.id}/{time_steps*(i+1)}")

env.close()
wandb.finish()

model.save("model_name")
task.upload_artifact("model", artifact_object="model_name.zip")
