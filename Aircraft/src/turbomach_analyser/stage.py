import numpy as np
from ..utils import (geometry as geom,
                     thermo)


class Stage:
    def __init__(self, is_compressor_stage, flow_coeff, work_coeff, axial_velocity, angular_velocity, hub_diameter, tip_diameter, reaction_init=0.5, diffusion_factor=None, lift_coeff=None):
        self.is_compressor_stage = is_compressor_stage
        self.flow_coeff = flow_coeff
        self.work_coeff = work_coeff
        self.angular_velocity = angular_velocity
        self.hub_diameter = hub_diameter
        self.tip_diameter = tip_diameter
        self.axial_velocity = axial_velocity
        self.reaction = reaction_init
        self.mean_radius = 0.5 * (tip_diameter + hub_diameter)
        self.blade_height = 0.5 * (tip_diameter - hub_diameter)
        self.mean_tangential_speed = geom.get_tangential_speed(
            angular_velocity, self.mean_radius * 2)
        self.d_stag_enthalpy = self.work_coeff * self.mean_tangential_speed**2
        if is_compressor_stage:
            self.alpha_1, self.alpha_2, self.alpha_3, self.beta_1, self.beta_2 = self.get_blade_angles()
            self.alpha_1_deg, self.alpha_2_deg, self.alpha_3_deg, self.beta_1_deg, self.beta_2_deg = np.rad2deg(
                np.array([self.alpha_1, self.alpha_2, self.alpha_3, self.beta_1, self.beta_2]))
        else:
            self.alpha_2, self.alpha_3, self.alpha_1, self.beta_2, self.beta_3 = self.get_blade_angles()
            self.alpha_2_deg, self.alpha_3_deg, self.alpha_1_deg, self.beta_2_deg, self.beta_3_deg = np.rad2deg(
                np.array([self.alpha_2, self.alpha_3, self.alpha_1, self.beta_2, self.beta_3]))
        # NOTE: consider setting first and last stage inlet/exit angles to 0

    def get_blade_angles(self):
        term_1 = 1 - self.reaction
        term_2 = self.d_stag_enthalpy / (2 * self.mean_tangential_speed**2)
        alpha_1 = np.arctan((term_1 - term_2) / self.flow_coeff)
        alpha_2 = np.arctan((term_1 + term_2) / self.flow_coeff)
        alpha_3 = 0  # NOTE: Check if this is true
        beta_1 = np.arctan((- self.reaction - term_2) / self.flow_coeff)
        beta_2 = np.arctan((- self.reaction + term_2) / self.flow_coeff)
        return alpha_1, alpha_2, alpha_3, beta_1, beta_2
