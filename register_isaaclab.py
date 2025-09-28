import gymnasium as gym

# 正确导入 IsaacLab 任务
try:
    from isaaclab_tasks import register_tasks
    print("Successfully imported isaaclab_tasks")
    register_tasks()
    print("Tasks registered successfully")
except ImportError as e:
    print(f"Import error: {e}")

# 查看注册的环境
from gymnasium import envs
isaac_envs = [env.id for env in envs.registry.values() if 'Isaac' in env.id]
print("Registered Isaac environments:", isaac_envs)

# 查看所有环境
all_envs = [env.id for env in envs.registry.values()]
print(f"Total registered environments: {len(all_envs)}")
print("First 10 environments:", all_envs[:10])
