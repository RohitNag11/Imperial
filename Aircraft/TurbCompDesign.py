import numpy as np


def get_core_diameter(engine_diameter, bypass_ratio):
    return 2 * ((bypass_ratio + (engine_diameter / 2)**2) / (bypass_ratio + 1))**0.5


def get_mean_radius(core_diameter, hub_tip_ratio):
    # TODO
    return core_diameter * (1 + hub_tip_ratio) / 4


def get_choke_axial_velocity(mass_flow, engine_diameter, bypass_ratio, hub_tip_ratio):
    # TODO
    return mass_flow / (np.pi * engine_diameter * (1 + bypass_ratio) * (1 + hub_tip_ratio) / 4)


def get_stagnation_enthalpy_change(C_P, static_temp_1, static_temp_2):
    return C_P * (static_temp_2 - static_temp_1)


comp_work_coeff = 0.5
comp_flow_coeff = 0.4  # phi
comp0_hub_tip_ratio = 0.35
FAN_HUB_TIP_RATIO = 0.35
C_P = 1.4
COMP_ENTRY_STAGNATION_TEMP = 249.75
COMP_EXIT_STAGNATION_TEMP = 758.17
MASS_FLOW = 164  # NOTE: check if corrected mass flow is required
BYPASS_RATIO = 0.5
ENGINE_DIAMETER = 1.5
CORE_DIAMETER = get_core_diameter(ENGINE_DIAMETER, BYPASS_RATIO)
OVERALL_STAGNATION_ENTHALPY_CHANGE = get_stagnation_enthalpy_change(
    C_P, COMP_ENTRY_STAGNATION_TEMP, COMP_EXIT_STAGNATION_TEMP)
OVERALL_STAGNATION_PRESSURE_CHANGE = get_stagnation_pressure_change(
    OVERALL_PRESSURE_RATIO, COMP_ENTRY_STATIC_PRESSURE)
mean_radius = get_mean_radius(CORE_DIAMETER, FAN_HUB_TIP_RATIO)
choke_axial_velocity = get_choke_axial_velocity(
    MASS_FLOW, ENGINE_DIAMETER, BYPASS_RATIO, FAN_HUB_TIP_RATIO)
axial_velocity = choke_axial_velocity * 0.85
u_mean = axial_velocity / 0.5
rotational_speed = u_mean / mean_radius
stage_static_enthalpy_change = comp_work_coeff * u_mean**2
stage_static_temp_change = stage_static_enthalpy_change / C_P
n_stages = OVERALL_STAGNATION_ENTHALPY_CHANGE / stage_static_enthalpy_change
