from abc import ABC, abstractmethod

class BaseEnv(ABC):
    @abstractmethod
    def __init__(self, device, distr_type, batch_size):
        self.type = distr_type
        self.device = device
        self.bs_per_horizon = int(batch_size)
    
    @abstractmethod
    def move_device(self, device):
        pass
    
    @abstractmethod
    def set_curriculum_params(self, param):
        pass

    @property
    @abstractmethod
    def curriculum_params(self):
        pass

    @property
    @abstractmethod
    def action_size(self):
        pass

    @abstractmethod
    def new_instance(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_reference_action(self):
        pass
    
    @abstractmethod
    def get_reward(self, action):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def plot_prob_figure(self, agent, pic_dir):
        pass

    @property
    def cnt_samples(self):
        return self.horizon * self.bs_per_horizon