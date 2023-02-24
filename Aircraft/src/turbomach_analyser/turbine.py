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
                 work_coefficient=2.2,
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
        self.is_low_pressure = is_low_pressure
        self.blade_length = kwargs['min_blade_length'] if 'min_blade_length' in kwargs else None
        # self.hub_diameter = kwargs['hub_diameter'] if 'hub_diameter' in kwargs else geom.get_hub_diameter_from_blade_length(
        #     kwargs['min_blade_length'], self.area_exit)
        self.mean_radius = geom.get_mean_radius_from_blade_length(
            kwargs['min_blade_length'], self.area_inlet) if 'min_blade_length' in kwargs else kwargs['mean_radius']
        self.d_stag_enthalpy = thermo.get_delta_stag_enthalpy(
            T0_inlet - T0_exit, SPEC_HEAT_CAPACITY)
        self.work_coefficient = work_coefficient
        self.angular_velocity = angular_velocity
        self.no_of_stages = int(np.ceil(self.__get_no_of_stages()))
        self.d_stag_temp_per_stage = (
            self.T0_inlet - self.T0_exit) / (self.no_of_stages-1)
        self.stag_temps = np.linspace(self.T0_inlet,
                                      self.T0_exit, self.no_of_stages + 1)
        isentropic_efficiency = 0.85
        self.pressure_ratios = 1 / (1 - self.d_stag_temp_per_stage/(isentropic_efficiency * 0.5 * (
            self.stag_temps[1:] + self.stag_temps[:-1])))**(SPEC_HEAT_RATIO / (SPEC_HEAT_RATIO - 1))
        self.hub_diameters, self.tip_diameters, self.hub_tip_ratios, self.areas, self.blade_lengths = self.__get_geometry_of_stages()

        # self.area_exit = 1
        # self.area_inlet = 1

    def __get_no_of_stages(self):

        # r_h = self.hub_diameter / 2
        # a = (r_h**2 + self.area_exit / np.pi)**0.5
        # b = (r_h**2 + self.area_inlet / np.pi)**0.5
        # # NOTE: c is very large, causes n_stages to converge to 1
        # c = self.work_coefficient * self.angular_velocity**2 / 4
        # d = a - b
        # e = r_h + b
        # f = 2 * (r_h+b)*(a-b)
        # g = (c*d**2)/3 + (c*e**2) + f
        # h = (c*d**2)/6 + (c*e**2) + f
        # n_stages = (h + (h**2 + 4 * g * self.d_stag_enthalpy)**0.5) / (2 * g)

        n_stages = self.d_stag_enthalpy / (self.work_coefficient * (
            self.area_inlet * self.angular_velocity / (2 * np.pi * self.blade_length))**2)
        return n_stages

    def __get_geometry_of_stages(self):
        inlet_hub_d = geom.get_hub_diameter_from_mean_radius(
            self.mean_radius, self.area_inlet)
        exit_hub_d = geom.get_hub_diameter_from_mean_radius(
            self.mean_radius, self.area_exit)
        inlet_tip_d = geom.get_tip_diameter_from_mean_radius(
            self.mean_radius, self.area_inlet)
        exit_tip_d = geom.get_tip_diameter_from_mean_radius(
            self.mean_radius, self.area_exit)
        hub_diameters = np.linspace(inlet_hub_d, exit_hub_d, self.no_of_stages)
        tip_diameters = np.linspace(inlet_tip_d, exit_tip_d, self.no_of_stages)
        hub_tip_ratios = hub_diameters / tip_diameters
        areas = geom.get_annulus_area(hub_tip_ratios, self.mean_radius)
        blade_lengths = (tip_diameters - hub_diameters) / 2
        return hub_diameters, tip_diameters, hub_tip_ratios, areas, blade_lengths
