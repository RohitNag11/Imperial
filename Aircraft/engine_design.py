import numpy as np
from src.turbomach_analyser import Engine
from src.utils import plots


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
        'hpt_work_coefficient': 1.0,
        'hpt_angular_velocity': 600,
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
    }


def main():
    consts = get_constants()
    engine_consts = get_engine_constants()
    engine = Engine(**engine_consts, **consts)
    compressors = {'LPC': engine.lpc, 'HPC': engine.hpc}
    turbines = {'HPT': engine.hpt, 'LPT': engine.lpt}

    print(f'fan tip diameter: {engine.fan.tip_diameter}')
    print(f'inner fan tip diameter:{engine.fan.inner_fan_tip_diameter}')
    print(f'fan hub diameter:{engine.fan.hub_diameter}')
    print(f'inner fan mean radius:{engine.fan.inner_fan_mean_radius}')
    print('****')
    for name, compressor in compressors.items():
        print(f'{name} tip diameter: {compressor.tip_diameters}')
        print(f'{name} hub diameter:{compressor.hub_diameters}')
        print(f'{name} mean radius:{compressor.mean_radius}')
        print(f'{name} hub-tip ratios:{compressor.hub_tip_ratios}')
        print(f'{name} blade lengths:{compressor.blade_lengths}')
        print(f'{name} annulus areas:{compressor.areas}')
        print('****')

    for name, turbine in turbines.items():
        print(f'{name} no of stages: {turbine.no_of_stages}')
        print(f'{name} inlet area: {turbine.area_inlet}')
        print(f'{name} exit area: {turbine.area_exit}')
        print(f'{name} angular velocity: {turbine.angular_velocity}')
        print(f'{name} mean radius: {turbine.mean_radius}')
        print(f'{name} flow coefficients: {turbine.flow_coefficients}')
        print(f'{name} tip mach nos: {turbine.tip_mach_nos}')
        print(f'{name} pressure ratios: {turbine.pressure_ratios}')
        print(f'{name} pressure ratio: {turbine.pressure_ratio}')
        print('****')
    print(
        f'turbine presure ratio: {np.prod(engine.hpt.pressure_ratios) * np.prod(engine.lpt.pressure_ratios)}')

    plots.draw_engine(engine)


if __name__ == '__main__':
    main()
