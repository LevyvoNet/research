import numpy as np

MAXIMUM_RAM = 1 * 2 ** 30  # Set maximum of 1 GB RAM
V_TYPE = np.float32
V_TYPE_SIZE = 8

from .id import IdPolicy
from .vi import ValueIterationPolicy,PrioritizedValueIterationPolicy
from .pi import PolicyIterationPolicy
from .rtdp import (RtdpPolicy)
from .ma_rtdp import MultiagentRtdpPolicy
# from .lrtdp import lrtdp
