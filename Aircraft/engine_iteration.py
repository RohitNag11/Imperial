import numpy as np
from src.turbomach_analyser import Engine
from src.utils import plots, formatter
import json
import itertools
import time
import os
from multiprocessing import Pool
from functools import partial


def get_constants():
    return {
        'SPEC_HEAT_RATIO': 1.4,
        'GAS_CONST': 287,
        'TEMP_SEA': 288.15,
        'SPEC_HEAT_CAPACITY': 1005
    }


def get_engine_constants():
    return {
        'mass_flow': 20.5,
        'bypass_ratio': 7,
        'overall_pressure_ratio': 40,
        'fan_hub_tip_ratio': 0.35,
        'fan_tip_mach_no': 1.3,
        'inner_fan_pressure_ratio': 1.8,
        'outer_fan_pressure_ratio': 2.5,
        'comp_axial_velocity': 190,
        'turbine_axial_velocity': 150,
        'turbine_isentropic_efficiency': 0.92,
        'lpc_pressure_ratio': 2.5,
        'per_stage_pressure_ratio': 1.3,
        'P_025': 91802,
        'T_025': 331.86,
        'P_03': 1468830,
        'T_03': 758.17,
        'P_044': 410468,
        'T_044': 1268.72,
        'P_045': 402258,
        'T_045': 1268.72,
        'P_05': 82688,
        'T_05': 892.91,
        'compressor_reaction_mean': 0.5,
        'compressor_reaction_tip': 0.5,
        'compressor_reaction_hub': 0.5,
        'turbine_reaction_mean': 0.5,
        'turbine_reaction_tip': 0.5,
        'turbine_reaction_hub': 0.5,
        'compressor_diffusion_factor': 0.45,
        'turbine_lift_coeff': 1,
    }


def __get_variable_ranges():
    return {
        'engine_diameter': np.linspace(2.5, 3.25, 4, endpoint=True),
        'lpt_work_coefficient': np.linspace(0.8, 2.4, 5, endpoint=True),
        'hpt_work_coefficient': np.linspace(0.8, 2, 5, endpoint=True),
        'hpt_angular_velocity': np.linspace(700, 1500, 5, endpoint=True),
        'min_blade_length': np.linspace(0.012, 0.015, 3, endpoint=True),
        'hpt_min_blade_length': np.linspace(0.012, 0.03, 5, endpoint=True),
        'lpt_min_blade_length': np.linspace(0.015, 0.05, 3, endpoint=True),
        # 'compressor_reaction_mean': np.linspace(0.2, 1, 4, endpoint=False),
        # 'compressor_diffusion_factor': np.linspace(2.5, 3.5, 1, endpoint=True),
        # 'turbine_reaction_mean': np.linspace(0.2, 1, 4, endpoint=False),
        # 'turbine_lift_coeff': np.linspace(2.5, 3.5, 1, endpoint=True),
    }


def get_number_of_iterations():
    d = __get_variable_ranges()
    return int(np.prod([len(v) for v in d.values()]))


def main(tried_vars_path, valid_vars_path):
    consts_kwargs = get_constants()
    engine_consts_kwargs = get_engine_constants()
    var_ranges_dict = __get_variable_ranges()
    all_possible_vars_dicts = ({k: v for k, v in zip(var_ranges_dict.keys(), p)}
                               for p in itertools.product(*var_ranges_dict.values()))
    if os.path.isfile(tried_vars_path):
        tried_var_key_hash, tried_var_vals_hash_set = formatter.csv_to_hashed_val_set(
            tried_vars_path)
    else:
        tried_var_key_hash = ''
        tried_var_vals_hash_set = set()

    if os.path.isfile(valid_vars_path):
        valid_var_key_hash, valid_var_vals_hash_set = formatter.csv_to_hashed_val_set(
            valid_vars_path)
    else:
        valid_var_key_hash = ''
        valid_var_vals_hash_set = set()

    no_iterations = get_number_of_iterations()
    print(f'No of iterations: {no_iterations}')
    var_key_hash = formatter.hash_dict_keys(var_ranges_dict)
    for var_dict in all_possible_vars_dicts:
        var_val_hash = formatter.hash_dict_vals(var_dict)
        if var_val_hash not in tried_var_vals_hash_set or var_key_hash != tried_var_key_hash:
            tried_var_vals_hash_set.add(var_val_hash)
            try:
                engine = Engine(**consts_kwargs,
                                **engine_consts_kwargs,
                                **var_dict)
                if engine.is_valid:
                    valid_var_vals_hash_set.add(var_val_hash)
            except:
                pass
    if var_key_hash != tried_var_key_hash:
        tried_var_key_hash = var_key_hash
        valid_var_key_hash = var_key_hash
    formatter.hashed_vals_to_csv(tried_var_key_hash,
                                 tried_var_vals_hash_set,
                                 tried_vars_path)
    formatter.hashed_vals_to_csv(valid_var_key_hash,
                                 valid_var_vals_hash_set,
                                 valid_vars_path)
    print(f'No of valid iterations: {len(valid_var_vals_hash_set)}')


if __name__ == '__main__':
    tried_variables_path = 'tried_variables_combinations.csv'
    valid_variables_path = 'valid_variables_combinations.csv'
    st = time.time()
    main(tried_variables_path, valid_variables_path)
    et = time.time()
    print(f'runtime: {formatter.format_elapsed_time(et - st)}')
