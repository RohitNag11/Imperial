import numpy as np

class Phase:
    def __init__(self, spec_heat_ratio, gas_const, density_sea, temp_sea, pressure_sea, **kwargs):
        self.mach_no = kwargs['mach_no'] if 'mach_no' in kwargs else self.__get_mach_no_from_velocity(
            kwargs['velocity_freestream'], spec_heat_ratio, gas_const, temp_sea)
        self.velocity_freestream = kwargs['velocity_freestream'] if 'velocity_freestream' in kwargs else self.__get_velocity_from_mach_no(
            kwargs['mach_no'], spec_heat_ratio, gas_const, temp_sea)
        self.coeff = kwargs['coeff'] if 'coeff' in kwargs else 1
        self.temp_ratio = kwargs['temp_ratio'] if 'temp_ratio' in kwargs else 1
        self.pressure_ratio = kwargs['pressure_ratio'] if 'pressure_ratio' in kwargs else 1
        self.density_ratio = kwargs['density_ratio'] if 'density_ratio' in kwargs else 1

        self.required_thrust = kwargs['required_thrust']
        self.specific_thrust = kwargs['specific_thrust']

        self.temp_freestream = self.__get_param_freestream(
            temp_sea, self.temp_ratio)
        self.pressure_freestream = self.__get_param_freestream(
            pressure_sea, self.pressure_ratio)
        self.density_freestream = self.__get_param_freestream(
            density_sea, self.density_ratio)

        self.mach_no_intake = self.mach_no * self.coeff
        self.temp_intake = self.__get_temp_intake(
            self.temp_freestream, self.mach_no_intake, spec_heat_ratio)
        self.pressure_intake = self.__get_pressure_intake(
            self.pressure_freestream, self.mach_no_intake, spec_heat_ratio)
        self.density_intake = self.__get_density_intake(
            self.density_freestream, self.mach_no_intake, spec_heat_ratio)
        self.velocity_intake = self.__get_velocity_from_mach_no(
            self.mach_no_intake, spec_heat_ratio, gas_const, self.temp_intake)

        if 'diameter' in kwargs:
            self.diameter = kwargs['diameter']
            self.area = self.__get_area_from_diameter(self.diameter)
            self.mass_flow = self.__get_mass_flow(
                self.velocity_intake, self.density_intake, self.area)
            self.mass_flow_corrected = self.__get_corrected_mass_flow(
                self.mass_flow, pressure_sea, temp_sea, self.pressure_intake, self.temp_intake)

        else:
            self.diameter, self.area, self.mass_flow, self.mass_flow_corrected = self.__get_min_engine_size(
                pressure_sea, temp_sea)

    def __get_mach_no_from_velocity(self, velocity, spec_heat_ratio, gas_const, temp_sea):
        return velocity / (spec_heat_ratio * gas_const * temp_sea)**0.5

    def __get_velocity_from_mach_no(self, mach_no, spec_heat_ratio, gas_const, temp):
        return mach_no * (spec_heat_ratio * gas_const * temp)**0.5

    def __get_param_freestream(self, param_sea, param_ratio):
        return param_sea * param_ratio

    def __get_temp_intake(self, temp_freestream, mach_no_intake, spec_heat_ratio):
        return temp_freestream / (1 + 0.5 * (spec_heat_ratio - 1) * mach_no_intake**2)

    def __get_pressure_intake(self, pressure_freestream, mach_no_intake, spec_heat_ratio):
        return pressure_freestream / ((1 + 0.5 * (spec_heat_ratio - 1) * mach_no_intake**2)**(spec_heat_ratio / (spec_heat_ratio - 1)))

    def __get_density_intake(self, density_freestream, mach_no_intake, spec_heat_ratio):
        return density_freestream / (1 + 0.5 * (spec_heat_ratio - 1) * mach_no_intake**2)**(1 / (spec_heat_ratio - 1))

    def __get_diameter_from_area(self, area):
        return 2 * (area / np.pi)**0.5

    def __get_area_from_diameter(self, diameter):
        return np.pi * (diameter / 2)**2

    def __get_mass_flow(self, velocity_intake, density_intake, area):
        return velocity_intake * density_intake * area

    def __get_corrected_mass_flow(self, mass_flow, pressure_sea, temp_sea, pressure_intake, temp_intake):
        return mass_flow * (pressure_sea / pressure_intake) * (temp_intake / temp_sea)**0.5

    def __get_min_engine_size(self, pressure_sea, temp_sea):
        min_mass_flow = self.required_thrust / self.specific_thrust
        min_area = min_mass_flow / (self.density_intake * self.velocity_intake)
        min_diameter = self.__get_diameter_from_area(min_area)
        min_mass_flow_corrected = self.__get_corrected_mass_flow(
            min_mass_flow, pressure_sea, temp_sea, self.pressure_intake, self.temp_intake)
        return min_diameter, min_area, min_mass_flow, min_mass_flow_corrected