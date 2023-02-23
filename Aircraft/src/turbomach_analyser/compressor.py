from .turbo_component import TurboComponent
from ..utils import geometry as geom
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
                 per_stage_pressure_ratio=1.3,
                 SPEC_HEAT_RATIO=1.4,
                 GAS_CONST=287,
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
        self.per_stage_pressure_ratio = per_stage_pressure_ratio
        self.no_of_stages = int(np.ceil(
            np.log(self.pressure_ratio) / np.log(per_stage_pressure_ratio)))
        self.mean_radius = geom.get_mean_radius_from_blade_length(
            kwargs['final_blade_length'], self.area_exit) if 'final_blade_length' in kwargs else kwargs['mean_radius']
        self.hub_diameters, self.tip_diameters, self.hub_tip_ratios, self.areas, self.blade_lengths = self.__get_geometry_of_stages()

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
