from pettingzoo.test import api_test, parallel_api_test, seed_test, parallel_seed_test, max_cycles_test, render_test, performance_benchmark, test_save_obs
from cell_env import cell_env


# API test
env = cell_env.env()
api_test(env)

# Parallel API test
env = cell_env.parallel_env()
parallel_api_test(env)

# Seed test
env_fn = cell_env.env
seed_test(env_fn)

parallel_env_fn = cell_env.parallel_env
parallel_seed_test(parallel_env_fn)

# Max Cycles test
max_cycles_test(cell_env)

# render test
render_test(env_fn)

# performance benchmark test
env = cell_env.env()
performance_benchmark(env)

# save observation test
test_save_obs(env)
