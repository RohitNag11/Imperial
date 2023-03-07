from .turbo_component import (TurboComponent)
from .stage import Stage
from ..utils import (geometry as geom,
                     thermo)
import numpy as np


class Compressor(TurboComponent):
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
                 reaction_mean=0.5,
                 reaction_tip=0.5,
                 reaction_hub=0.5,
                 SPEC_HEAT_RATIO=1.4,
                 GAS_CONST=287,
                 SPEC_HEAT_CAPACITY=1005,
                 ** kwargs):
        super().__init__(mass_flow,
                         axial_velocity,
                         pressure_ratio,
                         P0_exit,
                         T0_exit,
                         T0_inlet,
                         SPEC_HEAT_RATIO=SPEC_HEAT_RATIO,
                         GAS_CONST=GAS_CONST)
        self.is_low_pressure = is_low_pressure
        self.name = 'LPC' if is_low_pressure else 'HPC'
        self.per_stage_pressure_ratio = per_stage_pressure_ratio
        self.no_of_stages = int(np.ceil(
            np.log(self.pressure_ratio) / np.log(per_stage_pressure_ratio)))
        self.mean_radius = geom.get_mean_radius_from_blade_length(
            kwargs['final_blade_length'], self.area_exit) if 'final_blade_length' in kwargs else kwargs['mean_radius']
        self.angular_velocity = angular_velocity
        self.tangential_speed = geom.get_tangential_speed(
            angular_velocity, self.mean_radius * 2)
        self.hub_diameters, self.tip_diameters, self.hub_tip_ratios, self.areas, self.blade_lengths = self.__get_geometry_of_stages()
        # self.tip_mach_nos = self.__get_tip_mach_nos(SPEC_HEAT_RATIO, GAS_CONST)
        self.work_coeff = self.__get_work_coefficient(SPEC_HEAT_CAPACITY)
        self.flow_coeff = self.axial_velocity / self.tangential_speed
        self.stages = [Stage(is_compressor_stage=True,
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
        properties = {f'{self.name} tip diameter: {self.tip_diameters}',
                      f'{self.name} hub diameter:{self.hub_diameters}',
                      f'{self.name} mean radius:{self.mean_radius}',
                      f'{self.name} hub-tip ratios:{self.hub_tip_ratios}',
                      f'{self.name} blade lengths:{self.blade_lengths}',
                      f'{self.name} annulus areas:{self.areas}', }
        return self.name + super().__str__() + ':' + '\n' + '\n'.join(properties)

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

    def __get_work_coefficient(self, SPEC_HEAT_CAPACITY):
        return thermo.get_delta_stag_enthalpy(self.T0_exit - self.T0_inlet, SPEC_HEAT_CAPACITY) / (self.no_of_stages * self.tangential_speed ** 2)
