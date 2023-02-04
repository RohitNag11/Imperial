# author: Rohit Nag
# date: 02/02/2023

import numpy as np


class FlightAnalysis:
    def __init__(self, spec_heat_ratio=1.4, gas_const=287, density_sea=1.225, temp_sea=288.15, pressure_sea=101300):
        self.spec_heat_ratio = spec_heat_ratio
        self.gas_const = gas_const
        self.density_sea = density_sea
        self.temp_sea = temp_sea
        self.pressure_sea = pressure_sea
        self.phases: dict[str, Phase] = dict()

    def add_phase(self, phase_name, phase_conditions):
        phase = Phase(self.spec_heat_ratio,
                      self.gas_const,
                      self.density_sea,
                      self.temp_sea,
                      self.pressure_sea,
                      **phase_conditions)
        self.phases[phase_name] = phase


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


# Constants:
SPEC_HEAT_RATIO = 1.4
GAS_CONST = 287
DENSITY_SEA = 1.225
TEMP_SEA = 288.15
PRESSURE_SEA = 101300

# Phase paramters:
cruise_conditions = {
    'mach_no': 0.84,
    'required_thrust': 29900,
    'specific_thrust': 182,
    'density_ratio': 0.3,
    'temp_ratio': 0.75,
    'pressure_ratio': 0.22,
    'coeff': 0.6
}

top_of_climb_conditions = {
    'mach_no': 0.84,
    'required_thrust': 84000,
    'specific_thrust': 185,
    'density_ratio': 0.3,
    'temp_ratio': 0.75,
    'pressure_ratio': 0.22,
    'coeff': 0.6,
    'diameter': 2.6
}

take_off_normal_conditions = {
    'velocity_freestream': 70,
    'required_thrust': 116000,
    'specific_thrust': 236,
    'coeff': 1 / 0.6,
    'diameter': 2.6
}

take_off_failure_conditions = {
    'velocity_freestream': 80,
    'required_thrust': 232000,
    'specific_thrust': 270,
    'coeff': 1 / 0.6,
    'diameter': 2.6
}

flight_analysis = FlightAnalysis(
    SPEC_HEAT_RATIO, GAS_CONST, DENSITY_SEA, TEMP_SEA, PRESSURE_SEA)
flight_analysis.add_phase('cruise', cruise_conditions)
flight_analysis.add_phase('top_of_climb', top_of_climb_conditions)
flight_analysis.add_phase('take_off_normal', take_off_normal_conditions)
flight_analysis.add_phase('take_off_failure', take_off_failure_conditions)

print('CRUISE:')
cruise_phase = flight_analysis.phases['cruise']
print(f"Mach No: {cruise_phase.mach_no}")
print(f"Minimum Engine Diameter: {cruise_phase.diameter}")
print(f"Mass Flowrate: {cruise_phase.mass_flow}")
print(f"Corrected Mass Flowrate: {cruise_phase.mass_flow_corrected}")
print('\n')
print('TOP OF CLIMB:')
top_of_climb_phase = flight_analysis.phases['top_of_climb']
print(f"Mach No: {top_of_climb_phase.mach_no}")
print(f"Engine Diameter: {top_of_climb_phase.diameter}")
print(f"Mass Flowrate: {top_of_climb_phase.mass_flow}")
print(f"Corrected Mass Flowrate: {top_of_climb_phase.mass_flow_corrected}")
print('\n')
print('TAKE-OFF (Normal):')
take_off_normal_phase = flight_analysis.phases['take_off_normal']
print(f"Mach No: {take_off_normal_phase.mach_no}")
print(f"Engine Diameter: {take_off_normal_phase.diameter}")
print(f"Mass Flowrate: {take_off_normal_phase.mass_flow}")
print(f"Corrected Mass Flowrate: {take_off_normal_phase.mass_flow_corrected}")
print('\n')
print('TAKE-OFF (Engine Failure):')
take_off_failure_phase = flight_analysis.phases['take_off_failure']
print(f"Mach No: {take_off_failure_phase.mach_no}")
print(f"Engine Diameter: {take_off_failure_phase.diameter}")
print(f"Mass Flowrate: {take_off_failure_phase.mass_flow}")
print(f"Corrected Mass Flowrate: {take_off_failure_phase.mass_flow_corrected}")
