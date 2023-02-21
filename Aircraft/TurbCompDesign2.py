# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:40:39 2023

@author: dge18
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#Minor Functions

def get_core_diameter(engine_diameter, bypass_ratio):
    core_diameter = 2 * ((bypass_ratio + (engine_diameter / 2)**2) / (bypass_ratio + 1))**0.5
    return(core_diameter)


def get_mean_radius(core_diameter, hub_tip_ratio):
    mean_radius = core_diameter * (1 + hub_tip_ratio) / 4
    return(mean_radius)


def get_axial_velocity(mass_flow, diameter, rho):
    axial_velocity = mass_flow / (rho * np.pi * diameter**2 / 4)
    return(axial_velocity)


def get_stagnation_enthalpy_change(Cp, static_temp_1, static_temp_2):
    stagnation_enthalpy_change = Cp * (static_temp_2 - static_temp_1)
    return(stagnation_enthalpy_change)


def get_stagnation_pressure_change(OVERALL_PRESSURE_RATIO, COMP_ENTRY_STATIC_PRESSURE):
    stagnation_pressure_change = COMP_ENTRY_STATIC_PRESSURE * (OVERALL_PRESSURE_RATIO - 1)
    return(stagnation_pressure_change)


def get_exit_temperature_comp(pressure_ratio, T01, polyentropic_efficiency, gamma):
    T02 = T01 *(pressure_ratio) ** ((gamma - 1)/(polyentropic_efficiency*gamma))
    return(T02)
    
    
def get_exit_temperature_turb(pressure_ratio, T01, polyentropic_efficiency, gamma):
    T02 = T01 *(pressure_ratio) ** (polyentropic_efficiency*(gamma - 1)/(gamma))
    return(T02)




def plot_results(results, title, fig, axs):
    #Plot things
    fig.suptitle(title)
    axs[0, 0].plot(results['n'], results['A'])
    axs[0, 0].set_title('Area')
    axs[0, 1].plot(results['n'], results['Dt'])
    axs[0, 1].set_title('Dt')
    axs[0, 2].plot(results['n'], results['P0'])
    axs[0, 2].set_title('P0')    
    axs[1, 0].plot(results['n'], results['Dh'])
    axs[1, 0].set_title('Dh')
    axs[1, 1].plot(results['n'], results['T0'])
    axs[1, 1].set_title('T0')
    axs[1, 2].set_title('Lb')    



#~~~~~~~~~~~~~~~
#Major Functions
#~~~~~~~~~~~~~~~
    
def calculate_compressor(psi, phi, hub_r, pressure_ratio, P_entry, P0_exit, T0_entry, T0_exit, deltah0, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO):
    
    #Find choked mass flow rate
    mass_flow_choked = mass_flow/0.85
    
    A_stage_n = mass_flow_choked*((Cp*T0_exit)**(1/2))/(1.3*P0_exit)
    
    Dt_n = ((4 * A_stage_n) / (3.14*(1 - hub_r**2)))**(0.5)
    

    
    #Calcuate mean radius of the blade based on the hub to tip ratio of the final blade and its diameter
    mean_radius = get_mean_radius(Dt_n, hub_r)
    
    rho0_exit = P0_exit/(R * T0_exit)
    
    rho_exit = rho0_exit                #This isn't perfectly accurate but it is just to set an axial velocity which can be anything less than choke
    
    #Calcuate the axial velocity that will make the engine choke
    axial_velocity = get_axial_velocity(mass_flow, Dt_n, rho_exit)
    

    #Calcuate the velocity of the blade at the mean radius. This equation is given in the course task description
    u_mean = axial_velocity / 0.5
    
    #Convert the mean blade velocity to the roational speed.
    omega = u_mean / mean_radius
    
    
    
    #Calculate Change in Properties across each stage
    stage_static_enthalpy_change = psi * u_mean**2
    stage_static_temp_change = stage_static_enthalpy_change / Cp
    
    
    #Calcuate the number of stages needed for a given total change in enthalpy
    n_stages = int(deltah0 / stage_static_enthalpy_change)
    
    
    #Total change in pressure across the compressor
    deltaP0_total = P_entry * (pressure_ratio - 1)
    
    #Calcuate pressure drop across a single stage
    deltaP0 = deltaP0_total / n_stages
    
    
    #Define Lists for saving 
    P0_List = []
    T0_List = []
    rho0_List = []
    rho_List = []
    A_List = []
    Dt_List = []
    Dh_List = []
    Lb_List = []
    
    
    for n in np.arange(0, n_stages, 1):
        #Calculate Stagnation Pressure out of the stage
        if n == 0:
            P01 = P_entry
        else:
            P01 = P0_List[n-1]
        
        P02 = P01 + deltaP0
        P0_List.append(P02)
        
        
        #if P02/P01 > 1.3:
            #print('P02/P01 = ', P02/P01, ' for psi = ', psi, ', phi = ', phi, ', h0 = ', hub_r )
    
        
        
        #Calcuate Stagnation Temperature out of the stage
        if n == 0:
            T01 = T0_entry
        else:
            T01 = T0_List[n-1]
        
        T02 = get_exit_temperature_comp((P02/P01), T01, polyentropic_efficiency, gamma)
        T0_List.append(T02)
        
        
        #Calculate Stagnation Density out of the stage
        rho02 = P02/(R * T02)
        rho0_List.append(rho02)
        
        #Calcuate Axial Mach Number out of the stage
        Mz = axial_velocity/(gamma*R*T02)**(1/2)
    
        #Calculate static density out of the stage
        rho2 = rho02/(1 + (gamma - 1)/2 * Mz**2)**(1/(gamma - 1))
        rho_List.append(rho2)
        
        #Calculate Cross-Sectional Area
        A = mass_flow/ (rho2*axial_velocity)
        A_List.append(A)
        
        #Calculate Diameter of Stage
        deltaR = A/(4*3.14*mean_radius)
        Dt = 2 *(mean_radius + deltaR)
        Dh = 2 *(mean_radius - deltaR)
        
        Dt_List.append(Dt)
        Dh_List.append(Dh)
        
        Lb = (Dt - Dh)/2
        Lb_List.append(Lb)


    results = {
            'n': np.arange(0, n_stages, 1),
            'n_stages': n_stages,
            'P0': P0_List,
            'deltaP0': deltaP0,
            'T0': T0_List,
            'rho0': rho0_List,
            'rho': rho_List,
            'A': A_List,
            'Dt': Dt_List,
            'Dh': Dh_List,
            'Lb': Lb_List,
            'omega': omega
            }


    return(results)
    
    
    
    
    

def calculate_turbine(psi, phi, hub_r, pressure_ratio, P_entry, P0_entry, T0_entry, T0_exit, deltah0, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO):

    #Find choked mass flow rate
    mass_flow_choked = mass_flow/0.85
    
    A_stage_0 = mass_flow_choked*((Cp*T0_entry)**(1/2))/(1.3*P0_entry)
    
    Dt_0 = ((4 * A_stage_0) / (3.14*(1 - hub_r**2)))**(0.5)
    
    
    #Calcuate mean radius of the blade based on the hub to tip ratio of the final blade and its diameter
    mean_radius = get_mean_radius(Dt_0, hub_r)
    
    rho0_entry = P0_entry/(R * T0_entry)
    
    rho_entry = rho0_entry                #This isn't perfectly accurate but it is just to set an axial velocity which can be anything less than choke
    
    #Calcuate the axial velocity that will make the engine choke
    axial_velocity = get_axial_velocity(mass_flow, Dt_0, rho_entry)
    
    
    
      
    
    
    #Calcuate the velocity of the blade at the mean radius. This equation is given in the course task description
    u_mean = axial_velocity / 0.5
    
    #Convert the mean blade velocity to the roational speed.
    omega = u_mean / mean_radius
    
    
    #Calculate Change in Properties across each stage
    stage_static_enthalpy_change = psi * u_mean**2
    stage_static_temp_change = stage_static_enthalpy_change / Cp
    
    
    #Calcuate the number of stages needed for a given total change in enthalpy
    n_stages = int(-1* deltah0 / stage_static_enthalpy_change)
    
    deltaP0_total = P_entry * (1/pressure_ratio - 1)
        
    #Calcuate pressure drop across a single stage
    deltaP0 = deltaP0_total / n_stages
    
    #if deltaP0 > 1.3:
        #print('Error: deltaP0 >1.3 for psi = ', psi, ', phi = ', phi, ', h0 = ', hub_r )
    
    
    #Define Lists for saving 
    P0_List = []
    T0_List = []
    rho0_List = []
    rho_List = []
    A_List = []
    Dt_List = []
    Dh_List = []
    Lb_List = []
    
    
    for n in np.arange(0, n_stages, 1):
        #Calculate Stagnation Pressure out of the stage
        if n == 0:
            P01 = P_entry
        else:
            P01 = P0_List[n-1]
        
        P02 = P01 + deltaP0
        P0_List.append(P02)
        
        #Calcuate Stagnation Temperature out of the stage
        if n == 0:
            T01 = T0_entry
        else:
            T01 = T0_List[n-1]
        
        T02 = get_exit_temperature_turb((P02/P01), T01, polyentropic_efficiency, gamma)
        T0_List.append(T02)
        
        
        #Calculate Stagnation Density out of the stage
        rho02 = P02/(R * T02)
        rho0_List.append(rho02)
        
        #Calcuate Axial Mach Number out of the stage
        Mz = axial_velocity/(gamma*R*T02)**(1/2)
    
        #Calculate static density out of the stage
        rho2 = rho02/(1 + (gamma - 1)/2 * Mz**2)**(1/(gamma - 1))
        rho_List.append(rho2)
        
        #Calculate Cross-Sectional Area
        A = mass_flow/ (rho2*axial_velocity)
        A_List.append(A)
        
        #Calculate Diameter of Stage
        deltaR = A/(4*3.14*mean_radius)
        Dt = 2 *(mean_radius + deltaR)
        Dh = 2 *(mean_radius - deltaR)
        
        Dt_List.append(Dt)
        Dh_List.append(Dh)
        
        Lb = (Dt - Dh)/2
        Lb_List.append(Lb)

    
    results = {
            'n': np.arange(0, n_stages, 1),
            'n_stages': n_stages,
            'P0': P0_List,
            'deltaP0': deltaP0,
            'T0': T0_List,
            'rho0': rho0_List,
            'rho': rho_List,
            'A': A_List,
            'Dt': Dt_List,
            'Dh': Dh_List,
            'Lb': Lb_List,
            'omega': omega
            }


    return(results)
    
    
    
    
    
    



#~~~~~~~~~~~~~~~~~~
#General Properties
#~~~~~~~~~~~~~~~~~~


#Full Engine Thermo Properties
mass_flow = 164                         # NOTE: check if corrected mass flow is required
polyentropic_efficiency = 0.89          #NOTE: find actual value


#Air Constants
Cp = 1000      #J/kg
gamma = 1.4
R = 287         #J/(kgK)


#Engine Geometry
ENGINE_DIAMETER = 1.5
FAN_HUB_TIP_RATIO = 0.35
BYPASS_RATIO = 0.5
CORE_DIAMETER = get_core_diameter(ENGINE_DIAMETER, BYPASS_RATIO)




#~~~~~~~~~~~~~~~~~~~~~~~~~
#Turbomachinery Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~



#Low Pressure Compressor
T0_entry_comp_low = 249.75              #Kelvin
T0_exit_comp_low = 331.86               #Kelvin
pressure_ratio_comp_low = 2.5
P_entry_comp_low = 1.01*10**5           #NOTE: Get actual value   
P0_exit_comp_low = 91.8 * 10**4         #Note: Check values
deltah0_comp_low = get_stagnation_enthalpy_change(Cp, T0_entry_comp_low, T0_exit_comp_low)
deltaP0_comp_low = get_stagnation_pressure_change(deltah0_comp_low, P_entry_comp_low)




#High Pressure Compressor
T0_entry_comp_high = 331.86            #Kelvin
T0_exit_comp_high = 758.17
pressure_ratio_comp_high = 16
P_entry_comp_high = P_entry_comp_low * pressure_ratio_comp_low
P0_exit_comp_high = 1468 * 10**4        #Note: Check values
deltah0_comp_high = get_stagnation_enthalpy_change(Cp, T0_entry_comp_high, T0_exit_comp_high)
deltaP0_comp_high = get_stagnation_pressure_change(deltah0_comp_high, P_entry_comp_high)





#High Pressure Turbine
T0_entry_turb_high = 1677.7            #Kelvin
T0_exit_turb_high = 1268.72
pressure_ratio_turb_high = 16
P_entry_turb_high = P_entry_comp_high
P0_entry_turb_high = 1424 * 10**4         #Note: Check values
deltah0_turb_high = get_stagnation_enthalpy_change(Cp, T0_entry_turb_high, T0_exit_turb_high)
deltaP0_turb_high = get_stagnation_pressure_change(deltah0_turb_high, P_entry_turb_high)




#Low Pressure Turbine
T0_entry_turb_low = 1268.72            #Kelvin
T0_exit_turb_low = 892.91 
pressure_ratio_turb_low = 2.5
P_entry_turb_low = P_entry_comp_low    
P0_entry_turb_low = 402.25 * 10**4 
deltah0_turb_low = get_stagnation_enthalpy_change(Cp, T0_entry_turb_low, T0_exit_turb_low)
deltaP0_turb_low = get_stagnation_pressure_change(deltah0_turb_low, P_entry_turb_low)















#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Low Pressure Compressor Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

#Defining lists that will be used to graph 
psi_comp_low_list = []
phi_comp_low_list = []
hub_r_comp_high_list = []

for psi in np.arange(0.35, 0.5, 0.05):
    for phi in np.arange(0.4, 0.75, 0.05):
        for hub_r in np.arange(0.2, 0.95, 0.05):

            #Creating Lists for Plotting
            psi_comp_low_list.append(psi)
            phi_comp_low_list.append(phi)
            hub_r_comp_high_list.append(hub_r)
            
            #Calculating Compressor Values
            results_compressor_low = calculate_compressor(psi, phi, hub_r, pressure_ratio_comp_low, P_entry_comp_low, P0_exit_comp_low, T0_entry_comp_low, T0_exit_comp_low, deltah0_comp_low, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO)
            
            
            #Plotting
            plot_results(results_compressor_low, 'Low Pressure Compressor', fig, axs)
      
plt.show()   
                







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#High Pressure Compressor Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

#Defining lists that will be used to graph 
psi_comp_high_list = []
phi_comp_high_list = []
hub_r_comp_high_list = []

for psi in np.arange(0.35, 0.5, 0.05):
    for phi in np.arange(0.4, 0.75, 0.05):
        for hub_r in np.arange(0.2, 0.95, 0.05):

            #Creating Lists for Plotting
            psi_comp_high_list.append(psi)
            phi_comp_high_list.append(phi)
            hub_r_comp_high_list.append(hub_r)
            
            #Calculating Compressor Values
            results_compressor_high = calculate_compressor(psi, phi, hub_r, pressure_ratio_comp_high, P_entry_comp_high, P0_exit_comp_high, T0_entry_comp_high, T0_exit_comp_high, deltah0_comp_high, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO)
            
            #Plotting
            plot_results(results_compressor_high, 'High Pressure Compressor', fig, axs)
      
plt.show()   
                







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#High Pressure Turbine Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

#Defining lists that will be used to graph 
psi_turb_high_list = []
phi_turb_high_list = []
hub_r_turb_high_list = []

for psi in np.arange(0.35, 0.5, 0.05):
    for phi in np.arange(0.4, 0.75, 0.05):
        for hub_r in np.arange(0.2, 0.95, 0.05):

            #Creating Lists for Plotting
            psi_turb_high_list.append(psi)
            phi_turb_high_list.append(phi)
            hub_r_turb_high_list.append(hub_r)
            
            #Calculating Turbine Values
            results_turbine_high = calculate_turbine(psi, phi, hub_r, pressure_ratio_turb_high, P_entry_turb_high, P0_entry_turb_high, T0_entry_turb_high, T0_exit_turb_high, deltah0_turb_high, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO)
            

            
            
            #Plotting
            plot_results(results_turbine_high, 'High Pressure Turbine', fig, axs)
      
plt.show()   
                









#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Low Pressure Turbine Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

#Defining lists that will be used to graph 
psi_turb_low_list = []
phi_turb_low_list = []
hub_r_turb_high_list = []

for psi in np.arange(0.35, 0.5, 0.05):
    for phi in np.arange(0.4, 0.75, 0.05):
        for hub_r in np.arange(0.2, 0.95, 0.05):

            #Creating Lists for Plotting
            psi_turb_low_list.append(psi)
            phi_turb_low_list.append(phi)
            hub_r_turb_high_list.append(hub_r)
            
            #Calculating Turbine Values
            results_turbine_low = calculate_turbine(psi, phi, hub_r, pressure_ratio_turb_low, P_entry_turb_low, P0_entry_turb_low, T0_entry_turb_low, T0_exit_turb_low, deltah0_turb_low, mass_flow, polyentropic_efficiency, Cp, gamma, R, BYPASS_RATIO, FAN_HUB_TIP_RATIO)
            
            
            #Plotting
            plot_results(results_turbine_low, 'Low Pressure Turbine', fig, axs)
      
plt.show()   
                



















'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

mapable = ax.scatter(comp_work_coeff_list, comp_flow_coeff_list, A_List , c=comp0_hub_tip_ratio_list)
cbar = fig.colorbar(mapable, orientation='vertical')
        
ax.set_xlabel('Psi')
ax.set_ylabel('Phi')
ax.set_zlabel('Area')

cbar.set_label('temperature [K]')



plt.show()
'''














              
    

































