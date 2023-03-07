from .turbo_component import TurboComponent
from ..utils import (geometry as geom,
                     thermo)
from .stage import Stage
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
                 isentropic_efficiency=0.92,
                 work_coefficient=2.2,
                 reaction_mean=0.5,
                 reaction_tip=0.5,
                 reaction_hub=0.5,
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
        self.name = 'LPT' if is_low_pressure else 'HPT'
        self.blade_length = kwargs['min_blade_length'] if 'min_blade_length' in kwargs else None
        self.mean_radius = geom.get_mean_radius_from_blade_length(
            kwargs['min_blade_length'], self.area_inlet) if 'min_blade_length' in kwargs else kwargs['mean_radius']
        self.d_stag_enthalpy = thermo.get_delta_stag_enthalpy(
            T0_inlet - T0_exit, SPEC_HEAT_CAPACITY)
        self.work_coeff = work_coefficient
        self.angular_velocity = angular_velocity
        self.no_of_stages = int(np.ceil(self.__get_no_of_stages()))
        self.d_stag_temp_per_stage = (
            self.T0_inlet - self.T0_exit) / (self.no_of_stages-1)
        self.isentropic_efficiency = isentropic_efficiency
        self.pressure_ratios = self.__get_pressure_ratios(SPEC_HEAT_RATIO)
        self.hub_diameters, self.tip_diameters, self.hub_tip_ratios, self.areas, self.blade_lengths = self.__get_geometry_of_stages()
        self.pressure_ratio = np.prod(self.pressure_ratios)
        self.tip_mach_nos = self.__get_tip_mach_nos(SPEC_HEAT_RATIO, GAS_CONST)
        self.mean_tangential_speed = geom.get_tangential_speed(
            self.angular_velocity, self.mean_radius)
        self.flow_coeff = self.axial_velocity / self.mean_tangential_speed
        self.stages = [Stage(is_compressor_stage=False,
                             number=i + 1,
                             flow_coeff=self.flow_coeff,
                             work_coeff=self.work_coeff,
                             axial_velocity=self.axial_velocity,
                             angular_velocity=self.angular_velocity,
                             hub_diameter=self.hub_diameters[i],
                             tip_diameter=self.tip_diameters[i],
                             reaction_mean=reaction_mean,
                             reaction_hub=reaction_hub,
                             reaction_tip=reaction_tip) for i in range(self.no_of_stages)]

    def __str__(self):
        properties = {f'{self.name} no of stages: {self.no_of_stages}',
                      f'{self.name} no of stages: {self.no_of_stages}',
                      f'{self.name} inlet area: {self.area_inlet}',
                      f'{self.name} exit area: {self.area_exit}',
                      f'{self.name} angular velocity: {self.angular_velocity}',
                      f'{self.name} mean radius: {self.mean_radius}',
                      f'{self.name} flow coefficient: {self.flow_coeff}',
                      f'{self.name} tip mach nos: {self.tip_mach_nos}',
                      f'{self.name} pressure ratios: {self.pressure_ratios}',
                      f'{self.name} pressure ratio: {self.pressure_ratio}'}
        return self.name + super().__str__() + ':' + '\n' + '\n'.join(properties)

    def __get_no_of_stages(self):
        n_stages = self.d_stag_enthalpy / (self.work_coeff * (
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

    def __get_pressure_ratios(self, SPEC_HEAT_RATIO):
        stag_temps = np.linspace(self.T0_inlet,
                                 self.T0_exit,
                                 self.no_of_stages + 1)
        pressure_ratios = 1 / (1 - self.d_stag_temp_per_stage/(self.isentropic_efficiency * 0.5 * (
            stag_temps[1:] + stag_temps[:-1])))**(SPEC_HEAT_RATIO / (SPEC_HEAT_RATIO - 1))
        return pressure_ratios

    def __get_tip_mach_nos(self, SPEC_HEAT_RATIO, GAS_CONST):
        u_tips = geom.get_tangential_speed(
            self.angular_velocity, self.tip_diameters)
        stag_temps = np.linspace(self.T0_inlet,
                                 self.T0_exit, self.no_of_stages)
        static_temps = thermo.get_static_temp(
            stag_temps, self.axial_velocity, SPEC_HEAT_RATIO, GAS_CONST)
        speed_of_sounds = thermo.get_speed_of_sound(
            static_temps, SPEC_HEAT_RATIO, GAS_CONST)
        tip_mach_nos = u_tips / speed_of_sounds
        return tip_mach_nos
