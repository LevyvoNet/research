from gym_mapf.envs.utils import create_mapf_env

from available_solvers import *
from tests.performance_utils import (run_all,
                                     STRONG_SOLVERS,
                                     EXPERIMENT_SOLVERS)

BENCHMARKS = [
    {
        'env': create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:13;n_agents:2;fail_prob:0",
        'solver_describers': STRONG_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:12;n_agents:2;fail_prob:0",
        'solver_describers': STRONG_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1),
        'env_str': "map:room-32-32-4;scen:1;n_agents:2;fail_prob:0",
        'solver_describers': STRONG_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 12, 2, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:12;n_agents:2;fail_prob:0.1",
        'solver_describers': STRONG_SOLVERS
    },
    {
        'env': create_mapf_env('empty-8-8', 1, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-8-8;scen:1;n_agents:3;fail_prob:0.1; 2 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 2, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:2;n_agents:3;fail_prob:0.1;no conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('empty-8-8', 6, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-8-8;scen:6;n_agents:3;fail_prob:0.1",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 10, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:10;n_agents:3;fail_prob:0.1;no conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('room-32-32-4', 12, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:room-32-32-4;scen:12;n_agents:3;fail_prob:0.1;no conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('empty-16-16', 7, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-16-16;scen:7;n_agents:3;fail_prob:0.1;2 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('empty-16-16', 1, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-16-16;scen:1;n_agents:3;fail_prob:0.1;2 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS
    },
    {
        'env': create_mapf_env('empty-16-16', 1, 4, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-16-16;scen:1;n_agents:4;fail_prob:0.1;2 conflicts",
        'solver_describers': [long_ma_rtdp_min_describer]
    },
    {
        'env': create_mapf_env('empty-16-16', 1, 5, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-16-16;scen:1;n_agents:5;fail_prob:0.1;2 conflicts",
        'solver_describers': [long_ma_rtdp_min_describer]
    },
    {
        'env': create_mapf_env('empty-16-16', 1, 6, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-16-16;scen:1;n_agents:6;fail_prob:0.1;2 conflicts",
        'solver_describers': [long_ma_rtdp_min_describer]
    },
    {
        'env': create_mapf_env('sanity-1-8', None, 1, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-1-8;n_agents:1X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-2-8', None, 2, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-2-8;n_agents:2X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-3-8', None, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-3-8;n_agents:3X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-4-8', None, 4, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-4-8;n_agents:4X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-5-8', None, 5, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-5-8;n_agents:5X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-6-8', None, 6, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-6-8;n_agents:6X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-7-8', None, 7, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-7-8;n_agents:7X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-8-8', None, 8, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-8-8;n_agents:8X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-1-32', None, 1, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-1-32;n_agents:1X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-2-32', None, 2, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-2-32;n_agents:2X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-3-32', None, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-3-32;n_agents:3X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-4-32', None, 4, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-4-32;n_agents:4X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-5-32', None, 5, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-5-32;n_agents:5X1;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-2-8', None, 4, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-2-8;n_agents:2X2;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-3-8', None, 6, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-3-8;n_agents:3X2;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-4-8', None, 8, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-4-8;n_agents:4X2;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-1-16', None, 3, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-1-16;n_agents:1X3;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('sanity-2-16', None, 6, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:sanity-2-16;n_agents:2X3;fail_prob:0.1;0 conflicts",
        'solver_describers': EXPERIMENT_SOLVERS + [long_id_rtdp_describer]
    },
    {
        'env': create_mapf_env('empty-8-8', 6, 5, 0.1, 0.1, -1000, -1, -1),
        'env_str': "map:empty-8-8;scen:6;n_agents:5;fail_prob:0.1",
        'solver_describers': EXPERIMENT_SOLVERS
    },

]

if __name__ == '__main__':
    run_all(BENCHMARKS)
