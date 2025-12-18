from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .network.powergrid_env import GridMaintenanceEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["powergrid"] = partial(env_fn, env=GridMaintenanceEnv)