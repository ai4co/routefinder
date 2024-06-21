from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from routefinder.models.model import RouteFinderSingleVariantSampling

from .policy import MTPOMOPolicy

log = get_pylogger(__name__)


class MTPOMO(RouteFinderSingleVariantSampling):
    """
    Main MTPOMO model with single feature sampling at each batch
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: MTPOMOPolicy,
        **kwargs,
    ):
        super(MTPOMO, self).__init__(
            env,
            policy,
            **kwargs,
        )
