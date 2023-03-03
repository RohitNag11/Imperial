import numpy as np
from .compressor import Compressor
from .turbine import Turbine
from .fan import Fan


class Engine:
    def __init__(self,
                 mass_flow,
                 engine_diameter=2.6,
                 bypass_ratio=7,
                 overall_pressure_ratio=40,
                 fan_hub_tip_ratio=0.35,
                 fan_tip_mach_no=1.3,
                 inner_fan_pressure_ratio=1.8,
                 outer_fan_pressure_ratio=2.5,
                 comp_axial_velocity=190,
                 turbine_axial_velocity=150,
                 lpc_pressure_ratio=2.5,
                 per_stage_pressure_ratio=1.3,
                 lpt_work_coefficient=2,
                 hpt_work_coefficient=1.95,
                 hpt_angular_velocity=900,
                 turbine_isentropic_efficiency=0.9,
                 P_025=91802,
                 T_025=331.86,
                 P_03=1468830,
                 T_03=758.17,
                 P_041=1424765,
                 T_041=1677.70,
                 P_044=410468,
                 T_044=1268.72,
                 P_045=402258,
                 T_045=1268.72,
                 P_05=82688,
                 T_05=892.91,
                 min_blade_length=0.01,
                 lpt_min_blade_length=0.012,
                 GAS_CONST=1.4,
                 SPEC_HEAT_RATIO=287,
                 TEMP_SEA=288.15,
                 SPEC_HEAT_CAPACITY=1005):
        self.mass_flow = mass_flow
        self.engine_diameter = engine_diameter
        self.bypass_ratio = bypass_ratio
        self.overall_pressure_ratio = overall_pressure_ratio
        self.fan = Fan(engine_diameter=engine_diameter,
                       tip_mach_no=fan_tip_mach_no,
                       hub_tip_ratio=fan_hub_tip_ratio,
                       inner_fan_pressure_ratio=inner_fan_pressure_ratio,
                       outer_fan_pressure_ratio=outer_fan_pressure_ratio,
                       bypass_ratio=bypass_ratio,
                       SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                       GAS_CONST=GAS_CONST,
                       TEMP_SEA=TEMP_SEA)
        self.hpt = Turbine(is_low_pressure=False,
                           mass_flow=self.mass_flow,
                           axial_velocity=turbine_axial_velocity,
                           pressure_ratio=P_044/P_041,
                           P0_exit=P_044,
                           T0_exit=T_044,
                           T0_inlet=T_041,
                           angular_velocity=hpt_angular_velocity,
                           isentropic_efficiency=turbine_isentropic_efficiency,
                           work_coefficient=hpt_work_coefficient,
                           min_blade_length=0.02,
                           SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                           GAS_CONST=GAS_CONST,
                           SPEC_HEAT_CAPACITY=SPEC_HEAT_CAPACITY)
        self.lpt = Turbine(is_low_pressure=True,
                           mass_flow=self.mass_flow,
                           axial_velocity=turbine_axial_velocity,
                           pressure_ratio=P_05/P_045,
                           P0_exit=P_05,
                           T0_exit=T_05,
                           T0_inlet=T_045,
                           angular_velocity=self.fan.angular_velocity,
                           isentropic_efficiency=turbine_isentropic_efficiency,
                           work_coefficient=lpt_work_coefficient,
                           min_blade_length=lpt_min_blade_length,
                           SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                           GAS_CONST=GAS_CONST,
                           SPEC_HEAT_CAPACITY=SPEC_HEAT_CAPACITY)
        self.lpc = Compressor(is_low_pressure=True,
                              mass_flow=self.mass_flow,
                              axial_velocity=comp_axial_velocity,
                              pressure_ratio=lpc_pressure_ratio/inner_fan_pressure_ratio,
                              P0_exit=P_025,
                              T0_exit=T_025,
                              T0_inlet=self.__get_T_021(),
                              angular_velocity=self.fan.angular_velocity,
                              mean_radius=self.fan.inner_fan_mean_radius,
                              per_stage_pressure_ratio=per_stage_pressure_ratio,
                              SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                              GAS_CONST=GAS_CONST,
                              SPEC_HEAT_CAPACITY=SPEC_HEAT_CAPACITY)
        self.hpc = Compressor(is_low_pressure=False,
                              mass_flow=self.mass_flow,
                              axial_velocity=comp_axial_velocity,
                              pressure_ratio=overall_pressure_ratio/lpc_pressure_ratio,
                              P0_exit=P_03,
                              T0_exit=T_03,
                              T0_inlet=T_025,
                              angular_velocity=hpt_angular_velocity,
                              final_blade_length=min_blade_length,
                              per_stage_pressure_ratio=per_stage_pressure_ratio,
                              SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                              GAS_CONST=GAS_CONST,
                              SPEC_HEAT_CAPACITY=SPEC_HEAT_CAPACITY)
        return

    def __get_T_021(self):
        # TODO
        # thermo.get_static_temp
        return 260.73
