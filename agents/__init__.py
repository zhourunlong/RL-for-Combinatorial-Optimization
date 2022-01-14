REGISTRY = {}

from .sp_agent import SPAgent
REGISTRY["SP"] = SPAgent

from .okd_agent import OKDAgent
REGISTRY["OKD"] = OKDAgent
