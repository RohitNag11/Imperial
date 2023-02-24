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
        'lpc_pressure_ratio': 2.5,
        'per_stage_pressure_ratio': 1.3,
        'lpt_work_coefficient': 2.5,
        'hpt_work_coefficient': 1.2,
        'hpt_angular_velocity': 700,
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
        'min_blade_length': 0.012,
    }


def main():
    consts = get_constants()
    engine_consts = get_engine_constants()
    engine = Engine(**engine_consts, **consts)

    print(f'fan tip diameter: {engine.fan.tip_diameter}')
    print(f'inner fan tip diameter:{engine.fan.inner_fan_tip_diameter}')
    print(f'fan hub diameter:{engine.fan.hub_diameter}')
    print(f'inner fan mean radius:{engine.fan.inner_fan_mean_radius}')
    print('****')
    print(f'lpc tip diameter: {engine.lpc.tip_diameters}')
    print(f'lpc hub diameter:{engine.lpc.hub_diameters}')
    print(f'lpc mean radius:{engine.lpc.mean_radius}')
    print(f'lpc hub-tip ratios:{engine.lpc.hub_tip_ratios}')
    print(f'lpc blade lengths:{engine.lpc.blade_lengths}')
    print(f'lpc annulus areas:{engine.lpc.areas}')
    print('****')
    print(f'hpc tip diameter: {engine.hpc.tip_diameters}')
    print(f'hpc hub diameter:{engine.hpc.hub_diameters}')
    print(f'hpc mean radius:{engine.hpc.mean_radius}')
    print(f'hpc hub-tip ratios:{engine.hpc.hub_tip_ratios}')
    print(f'hpc blade lengths:{engine.hpc.blade_lengths}')
    print(f'hpc annulus areas:{engine.hpc.areas}')
    print('****')
    print(f'hpt no of stages: {engine.hpt.no_of_stages}')
    print(f'hpt inlet area: {engine.hpt.area_inlet}')
    print(f'hpt exit area: {engine.hpt.area_exit}')
    print(f'hpt angular velocity: {engine.hpt.angular_velocity}')
    print(f'hpt mean radius: {engine.hpt.mean_radius}')
    print(f'hpt pressure_ratios: {engine.hpt.pressure_ratios}')
    print('****')
    print(f'lpt no of stages: {engine.lpt.no_of_stages}')
    print(f'lpt inlet area: {engine.lpt.area_inlet}')
    print(f'lpt exit area: {engine.lpt.area_exit}')
    print(f'lpt angular velocity: {engine.lpt.angular_velocity}')
    print(f'lpt mean radius: {engine.lpt.mean_radius}')
    print(f'lpt pressure_ratios: {engine.lpt.pressure_ratios}')

    plots.draw_engine(engine)


if __name__ == '__main__':
    main()
