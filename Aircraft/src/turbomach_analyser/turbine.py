from .turbo_component import TurboComponent
from ..utils import (geometry as geom,
                     thermo)
import numpy as np


class Turbine(TurboComponent):
    def __init__(self,
                 is_low_pressure,
                 mass_flow,
                 axial_velocity,
                 pressure_ratio,
                 P0_exit,
                 T0_exit,
                 T0_inlet,
                 angular_velocity,
                 per_stage_pressure_ratio=1.3,
                 work_coefficient=0.9,
                 SPEC_HEAT_RATIO=1.4,
                 GAS_CONST=287,
                 SPEC_HEAT_CAPACITY=1005,
                 **kwargs):
        super().__init__(mass_flow,
                         axial_velocity,
                         pressure_ratio,
                         P0_exit,
                         T0_exit,
                         T0_inlet,
                         SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                         GAS_CONST=GAS_CONST)
        self.area_inlet = 1
        self.area_exit = 4
        self.is_low_pressure = is_low_pressure
        self.hub_diameter = kwargs['hub_diameter'] if 'hub_diameter' in kwargs else geom.get_hub_diameter_from_blade_length(
            kwargs['min_blade_length'], self.area_exit)
        self.d_stag_enthalpy = thermo.get_delta_stag_enthalpy(
            T0_inlet - T0_exit, SPEC_HEAT_CAPACITY)
        self.work_coefficient = work_coefficient
        self.angular_velocity = angular_velocity
        self.no_of_stages = self.__get_no_of_stages()  # NOTE: Might be wrong

    def __get_no_of_stages(self):
        inlet_tip_d = geom.get_tip_diameter_from_mean_radius(
            self.hub_diameter, self.area_inlet)
        exit_tip_d = geom.get_tip_diameter_from_mean_radius(
            self.hub_diameter, self.area_exit)
        r_tn = exit_tip_d / 2
        r_t0 = inlet_tip_d / 2
        r_h = self.hub_diameter / 2
        c = self.d_stag_enthalpy / \
            (self.work_coefficient * self.angular_velocity**2)
        radius_sum = r_tn + r_t0 + 2*r_h
        n_stages = (-r_tn + r_t0 + ((r_t0 - r_tn)**2 + 16 *
                    c * radius_sum)**0.5)/(2*radius_sum)
        return n_stages
