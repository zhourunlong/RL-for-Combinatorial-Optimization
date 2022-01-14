REGISTRY = {}

from .sp_env import SPEnv
REGISTRY["SP"] = SPEnv

from .okd_env import OKDEnv
REGISTRY["OKD"] = OKDEnv