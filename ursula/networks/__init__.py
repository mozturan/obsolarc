from .base import BaseNetwork,BasePolicy, BaseValueFunction
from .gaussianactor import GaussianActor
from .critic import Critic
# from .transformer import TransformerActor # Future addition

# "Menu"
ACTOR_REGISTRY = {
    "gaussian": GaussianActor,
    # "transformer": TransformerActor,
}

CRITIC_REGISTRY = {
    "mlp": Critic,
}