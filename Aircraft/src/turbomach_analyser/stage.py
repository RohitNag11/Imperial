import numpy as np
from ..utils import (geometry as geom,
                     thermo)


class Stage:
    def __init__(self,
                 is_compressor_stage,
                 number,
                 flow_coeff,
                 work_coeff,
                 axial_velocity,
                 angular_velocity,
                 hub_diameter,
                 tip_diameter,
                 reaction_mean=0.5,
                 reaction_hub=0.5,
                 reaction_tip=0.5,
                 diffusion_factor=None,
                 lift_coeff=None):
        self.number = number
        self.is_compressor_stage = is_compressor_stage
        self.flow_coeff = flow_coeff
        self.work_coeff = {}
        self.work_coeff['mean'] = work_coeff
        self.angular_velocity = angular_velocity
        self.hub_diameter = hub_diameter
        self.tip_diameter = tip_diameter
        self.axial_velocity = axial_velocity
        self.reaction = {'mean': reaction_mean,
                         'hub': reaction_hub,
                         'tip': reaction_tip}
        self.mean_radius = 0.5 * (tip_diameter + hub_diameter)
        self.blade_height = 0.5 * (tip_diameter - hub_diameter)
        self.mean_tangential_speed = geom.get_tangential_speed(
            angular_velocity, self.mean_radius * 2)
        self.d_stag_enthalpy = work_coeff * self.mean_tangential_speed**2
        self.blade_angles_rad = {}
        # NOTE: consider setting first and last stage inlet/exit angles to 0
        self.blade_angles_rad['mean'] = self.get_mean_blade_angles()
        self.blade_angles_rad['hub'] = self.get_blade_angles_at_radius(radius=hub_diameter / 2,
                                                                       reaction=reaction_hub)
        self.blade_angles_rad['tip'] = self.get_blade_angles_at_radius(radius=tip_diameter / 2,
                                                                       reaction=reaction_tip)

        # self.mean_blade_angles = self.get_mean_blade_angles()
        # self.hub_blade_angles = self.get_blade_angles_at_radius(radius=hub_diameter / 2,
        #                                                         reaction=reaction_hub,
        #                                                         label='hub')
        # self.tip_blade_angles = self.get_blade_angles_at_radius(radius=tip_diameter / 2,
        #                                                         reaction=reaction_tip,
        #                                                         label='tip')
        self.work_coeff['hub'] = self.get_work_coeff_at_location('hub')
        self.work_coeff['tip'] = self.get_work_coeff_at_location('tip')
        self.blade_angles_deg = {}
        self.blade_angles_deg['mean'] = {key: np.rad2deg(
            val) for key, val in self.blade_angles_rad['mean'].items()}
        self.blade_angles_deg['hub'] = {key: np.rad2deg(
            val) for key, val in self.blade_angles_rad['hub'].items()}
        self.blade_angles_deg['tip'] = {key: np.rad2deg(
            val) for key, val in self.blade_angles_rad['tip'].items()}

    def get_mean_blade_angles(self):
        reaction = self.reaction['mean']
        term_1 = 1 - reaction
        term_2 = self.d_stag_enthalpy / (2 * self.mean_tangential_speed**2)
        alpha_a = np.arctan((term_1 - term_2) / self.flow_coeff)
        alpha_b = np.arctan((term_1 + term_2) / self.flow_coeff)
        alpha_c = 0  # NOTE: Check if this is true
        beta_a = np.arctan((- reaction - term_2) / self.flow_coeff)
        beta_b = np.arctan((- reaction + term_2) / self.flow_coeff)
        alpha_a_label = 'alpha_1' if self.is_compressor_stage else 'alpha_2'
        alpha_b_label = 'alpha_2' if self.is_compressor_stage else 'alpha_3'
        alpha_c_label = 'alpha_3' if self.is_compressor_stage else 'alpha_1'
        beta_a_label = 'beta_1' if self.is_compressor_stage else 'beta_2'
        beta_b_label = 'beta_2' if self.is_compressor_stage else 'beta_3'
        return {
            alpha_a_label: alpha_a,
            alpha_b_label: alpha_b,
            alpha_c_label: alpha_c,
            beta_a_label: beta_a,
            beta_b_label: beta_b
        }

    def get_blade_angles_at_radius(self, radius, reaction):
        alpha_2_mean = self.blade_angles_rad['mean']['alpha_2']
        alpha_2_r = np.arctan(np.tan(alpha_2_mean) *
                              (radius / self.mean_radius))
        alpha_3_r = np.arctan(
            ((2 * (1 - reaction)) / self.flow_coeff) - np.tan(alpha_2_mean))
        return {
            f'alpha_2': alpha_2_r,
            f'alpha_3': alpha_3_r
        }

    def get_work_coeff_at_location(self, location):
        alpha_3_r = self.blade_angles_rad[location]['alpha_3']
        alpha_2_r = self.blade_angles_rad[location]['alpha_2']
        return self.flow_coeff * (np.tan(alpha_3_r) - np.tan(alpha_2_r))
