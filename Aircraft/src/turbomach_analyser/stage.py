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
        self.lift_coeff = lift_coeff if lift_coeff else None
        self.diffusion_factor = diffusion_factor if diffusion_factor else None
        self.flow_coeff = {}
        self.flow_coeff['mean'] = flow_coeff
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
        self.blade_angles_rad['mean'] = self.get_blade_angles_at_mean_radius2()
        # self.blade_angles_rad['hub'] = self.get_blade_angles_at_radius(radius=hub_diameter / 2,
        #                                                                reaction=reaction_hub)
        # self.blade_angles_rad['tip'] = self.get_blade_angles_at_radius(radius=tip_diameter / 2,
        #                                                                reaction=reaction_tip)

        # self.mean_blade_angles = self.get_mean_blade_angles()
        # self.hub_blade_angles = self.get_blade_angles_at_radius(radius=hub_diameter / 2,
        #                                                         reaction=reaction_hub,
        #                                                         label='hub')
        # self.tip_blade_angles = self.get_blade_angles_at_radius(radius=tip_diameter / 2,
        #                                                         reaction=reaction_tip,
        #                                                         label='tip')

        # self.work_coeff['hub'] = self.get_work_coeff_at_location('hub')
        # self.work_coeff['tip'] = self.get_work_coeff_at_location('tip')
        self.blade_angles_deg = {}
        self.blade_angles_deg['mean'] = {key: np.rad2deg(
            val) for key, val in self.blade_angles_rad['mean'].items()}
        # self.blade_angles_deg['hub'] = {key: np.rad2deg(
        #     val) for key, val in self.blade_angles_rad['hub'].items()}
        # self.blade_angles_deg['tip'] = {key: np.rad2deg(
        #     val) for key, val in self.blade_angles_rad['tip'].items()}
        self.solidity = self.get_solidity()
        self.aspect_ratio = 1
        self.no_of_blades = self.get_no_of_blades()
        self.is_valid = self.__check_validity()

    def get_blade_angles_at_mean_radius(self):
        # make phi and si negative if compressor.
        flow_coeff = self.flow_coeff['mean']
        work_coeff = self.work_coeff['mean']
        reaction = self.reaction['mean']
        term_1 = (work_coeff + 2 * reaction) / (2 * flow_coeff)
        term_2 = (work_coeff - 2 * reaction) / (2 * flow_coeff)
        if self.is_compressor_stage:
            return {
                'alpha_1': np.arctan((1 / flow_coeff) - term_1),
                'alpha_2': np.arctan((1 / flow_coeff) + term_2),
                'beta_1': np.arctan(-term_1),
                'beta_2': np.arctan(term_2)
            }
        return {
            'alpha_2': np.arctan((1 / flow_coeff) + term_2),
            'alpha_3': np.arctan((1 / flow_coeff) - term_1),
            'beta_2': np.arctan(term_2),
            'beta_3': np.arctan(-term_1)
        }

    def get_blade_angles_at_mean_radius2(self):
        z = self.axial_velocity
        u = self.mean_tangential_speed
        cz_2u = 0.5*z/u
        r = self.reaction['mean']
        dh = self.d_stag_enthalpy
        R = np.array([r, r-1, dh, dh])
        if self.is_compressor_stage:
            C = np.array([[0, 0, cz_2u, cz_2u],
                          [-cz_2u, -cz_2u, 0, 0],
                          [0, 0, u*z, -u*z],
                          [-u*z, u*z, 0, 0]])
        else:
            C = np.array([[0, 0, cz_2u, cz_2u],
                          [-cz_2u, -cz_2u, 0, 0],
                          [0, 0, -u*z, u*z],
                          [u*z, -u*z, 0, 0]])
        T = np.arctan(np.linalg.solve(C, R))
        if self.is_compressor_stage:
            return {
                'alpha_1': T[0],
                'alpha_2': T[1],
                'beta_1': T[2],
                'beta_2': T[3]
            }
        return {
            'alpha_2': T[0],
            'alpha_3': T[1],
            'beta_2': T[2],
            'beta_3': T[3]
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

    def get_solidity(self):
        inlet = 'alpha_1' if self.is_compressor_stage else 'alpha_2'
        outlet = 'alpha_2' if self.is_compressor_stage else 'alpha_3'
        alpha_in = self.blade_angles_rad['mean'][inlet]
        alpha_out = self.blade_angles_rad['mean'][outlet]
        c1 = np.cos(alpha_in)
        c2 = np.cos(alpha_out)
        t2 = np.tan(alpha_out)
        t1 = np.tan(alpha_in)
        if self.is_compressor_stage:
            df = self.diffusion_factor
            return (c1 * c2 * (t2 - t1)) / (2 * (c1 - c2 * (1 - df)))
        z = self.lift_coeff
        return np.abs(z / (2 * c2**2 * (t2 - t1)))

    def get_no_of_blades(self):
        return 2 * np.pi * self.mean_radius * self.aspect_ratio / self.blade_height

    def __check_validity(self):
        # blade heights cannot be below 10mm
        if self.blade_height < 0.01:
            return False
        return True
