import numpy as np
from src.turbomach_analyser import Engine
from src.utils import plots, formatter
import json


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
        'engine_diameter': 2.6,
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
        'lpt_work_coefficient': 2.5,
        'hpt_work_coefficient': 0.8,
        'hpt_angular_velocity': 1250,
        'min_blade_length': 0.012,
        'lpt_min_blade_length': 0.03,
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
        'compressor_diffusion_factor': 0.45,
        'turbine_reaction_mean': 0.5,
        'turbine_reaction_tip': 0.5,
        'turbine_reaction_hub': 0.5,
        'turbine_lift_coeff': 0.8
    }


def main():
    engine = Engine(**get_constants(), **get_engine_constants())
    components = [engine.fan, engine.lpc, engine.hpc, engine.hpt, engine.lpt]
    [print(f'{component}\n*****\n') for component in components]
    formatter.save_obj_to_file(engine, 'Aircraft/engine_design.json')
    plots.draw_engine(engine)


if __name__ == '__main__':
    main()
