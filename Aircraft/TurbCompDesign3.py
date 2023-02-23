# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:40:39 2023

@author: dge18
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#Minor Functions


def get_mean_radius(diameter, hub_tip_ratio):
    mean_radius = diameter * (1 + hub_tip_ratio) / 4
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


def get_static_pressure(stagnation_pressure, M, gamma):
    static_pressure = stagnation_pressure*(1 - (gamma - 1)*(M**2)/2 )**(-gamma/(gamma - 1))
    return(static_pressure)



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
    axs[1, 2].plot(results['n'], results['Lb'])
    axs[1, 2].set_title('Lb')    


'''
def plot_results_3D(n, omega, hub_r, psi, phi, title, fig, ax):
    mapable = ax.scatter(omega, hub_r, n , c=phi)
    cbar = fig.colorbar(mapable, orientation='vertical')
            
    ax.set_xlabel('Hub_to_tip_ratio')
    ax.set_ylabel('omega')
    ax.set_zlabel('n')
    
    cbar.set_label('phi')

'''


#~~~~~~~~~~~~~~~
#Major Functions
#~~~~~~~~~~~~~~~
    
def calculate_compressor(omega, hub_r, pressure_ratio, P_entry, P0_entry, P0_exit, T0_entry, T0_exit, pressure_ratio_stage, mass_flow, polyentropic_efficiency, Cp, gamma, R):
    
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
    u_mean = omega * mean_radius
    
    
    
    #Calculate Overall enthalpy change
    deltah0_overall = Cp*(T0_exit - T0_entry)
    
    
    
    #Calcuate the number of stages needed for a given total change in enthalpy
    n_stages = int(np.ceil(pressure_ratio / pressure_ratio_stage))
    
    pressure_ratio_stage = pressure_ratio / n_stages            #this eliminates the rounding issue
    
    
    #Calcuate Phi
    phi = axial_velocity / u_mean
    
    
    #Define Lists for saving 
    P0_List = []
    T0_List = []
    h0_List = []
    psi_List = []
    rho0_List = []
    rho_List = []
    A_List = []
    Dt_List = []
    Dh_List = []
    Lb_List = []
    
    
    for n in np.arange(0, n_stages + 1, 1):
        #Calculate Stagnation Pressure out of the stage
        if n == 0:
            P01 = P_entry
        else:
            P01 = P0_List[n-1]
        
        P02 = P01 * (pressure_ratio_stage + 1)
        P0_List.append(P02)
        
        
        #Calcuate Stagnation Temperature out of the stage
        if n == 0:
            T01 = T0_entry
        else:
            T01 = T0_List[n-1]
        
        T02 = get_exit_temperature_comp((P02/P01), T01, polyentropic_efficiency, gamma)
        T0_List.append(T02)
        
        
        if n == 0:
            h02 = Cp*(T01)
        else:
            h02 = h0_List[n-1] + Cp*(T02 - T01)
        
        h0_List.append(h02)
        
        
        psi = Cp*(T02 - T01)/(u_mean**2)
        psi_List.append(psi)
        
        
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
            'n': np.arange(0, n_stages + 1),
            'n_stages': n_stages,
            'P0': P0_List,
            'deltaP0': pressure_ratio_stage,
            'T0': T0_List,
            'rho0': rho0_List,
            'rho': rho_List,
            'A': A_List,
            'Dt': Dt_List,
            'Dh': Dh_List,
            'Lb': Lb_List,
            'psi': psi_List,
            'psiMean': np.mean(psi_List),
            'omega': omega,
            'phi': phi
            }


    return(results)
    
    
    
    
    

def calculate_turbine(omega, hub_r, pressure_ratio, P_entry, P0_entry, P0_exit, T0_entry, T0_exit, pressure_ratio_stage, mass_flow, polyentropic_efficiency, Cp, gamma, R):

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
    u_mean = omega * mean_radius
    
    #Calculate Overall enthalpy change
    deltah0_overall = Cp*(T0_exit - T0_entry)
    
    
    
    #Calcuate the number of stages needed for a given total change in enthalpy
    n_stages = int(np.ceil(pressure_ratio / pressure_ratio_stage))
    
    pressure_ratio_stage = pressure_ratio / n_stages            #this eliminates the rounding issue
    
    #Calcuate Phi
    phi = axial_velocity / u_mean
    
    
    
    
    
    #Define Lists for saving 
    P0_List = []
    T0_List = []
    h0_List = []
    psi_List = []
    rho0_List = []
    rho_List = []
    A_List = []
    Dt_List = []
    Dh_List = []
    Lb_List = []
    
    
    for n in np.arange(0, n_stages + 1, 1):
        #Calculate Stagnation Pressure out of the stage
        if n == 0:
            P01 = P_entry
        else:
            P01 = P0_List[n-1]
        
        P02 = P01 * (pressure_ratio_stage + 1)
        P0_List.append(P02)
        
        #Calcuate Stagnation Temperature out of the stage
        if n == 0:
            T01 = T0_entry
        else:
            T01 = T0_List[n-1]
        
        T02 = get_exit_temperature_turb((P02/P01), T01, polyentropic_efficiency, gamma)
        T0_List.append(T02)
        
        
        if n == 0:
            h02 = Cp*(T01)
        else:
            h02 = h0_List[n-1] + Cp*(T02 - T01)
        
        h0_List.append(h02)
        
        
        psi = Cp*(T02 - T01)/(u_mean**2)
        psi_List.append(psi)
        
        
        
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
            'n': np.arange(0, n_stages + 1),
            'n_stages': n_stages,
            'P0': P0_List,
            'deltaP0': pressure_ratio_stage,
            'T0': T0_List,
            'rho0': rho0_List,
            'rho': rho_List,
            'A': A_List,
            'Dt': Dt_List,
            'Dh': Dh_List,
            'Lb': Lb_List,
            'psi': psi_List,
            'psiMean': np.mean(psi_List),
            'omega': omega,
            'phi': phi
            }


    return(results)
    
    
    
    



#~~~~~~~~~~~~~~~~~~
#General Properties
#~~~~~~~~~~~~~~~~~~


#Full Engine Thermo Properties
mass_flow = 164                         # NOTE: check if corrected mass flow is required
polyentropic_efficiency = 0.89          #NOTE: find actual value
pressure_ratio_comp_max = 1.3
pressure_ratio_turb_max = 2.5


#Air Constants
Cp = 1000      #J/kg
gamma = 1.4
R = 287         #J/(kgK)


#Engine Geometry
ENGINE_DIAMETER = 1.5
FAN_HUB_TIP_RATIO = 0.35
BYPASS_RATIO = 0.5





#~~~~~~~~~~~~~~~~~~~~~~~~~
#Turbomachinery Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~



#Low Pressure Compressor
T0_entry_comp_low = 249.75              #Kelvin
T0_exit_comp_low = 331.86               #Kelvin
pressure_ratio_comp_low = 2.5
P0_entry_comp_low = 36.721 * 10**3
Mz_entry_comp_low = 1.1
P_entry_comp_low = get_static_pressure(P0_entry_comp_low, Mz_entry_comp_low, gamma)
P0_exit_comp_low = 91.8 * 10**3         #Note: Check values
deltah0_comp_low = get_stagnation_enthalpy_change(Cp, T0_entry_comp_low, T0_exit_comp_low)
deltaP0_comp_low = get_stagnation_pressure_change(deltah0_comp_low, P_entry_comp_low)




#High Pressure Compressor
T0_entry_comp_high = 331.86            #Kelvin
T0_exit_comp_high = 758.17
pressure_ratio_comp_high = 16
P_entry_comp_high = P_entry_comp_low * pressure_ratio_comp_low
P0_entry_comp_high = 91.8 * 10**3
P0_exit_comp_high = 1468 * 10**3        #Note: Check values
deltah0_comp_high = get_stagnation_enthalpy_change(Cp, T0_entry_comp_high, T0_exit_comp_high)
deltaP0_comp_high = get_stagnation_pressure_change(deltah0_comp_high, P_entry_comp_high)





#High Pressure Turbine
T0_entry_turb_high = 1677.7            #Kelvin
T0_exit_turb_high = 1268.72
pressure_ratio_turb_high = 16
P_entry_turb_high = P_entry_comp_high
P0_entry_turb_high = 1424 * 10**3         #Note: Check values
P0_exit_turb_high = 410.4 * 10**3           #Note: Check values
deltah0_turb_high = get_stagnation_enthalpy_change(Cp, T0_entry_turb_high, T0_exit_turb_high)
deltaP0_turb_high = get_stagnation_pressure_change(deltah0_turb_high, P_entry_turb_high)




#Low Pressure Turbine
T0_entry_turb_low = 1268.72            #Kelvin
T0_exit_turb_low = 892.91 
pressure_ratio_turb_low = 2.5
P_entry_turb_low = P_entry_comp_low    
P0_entry_turb_low = 402.25 * 10**3 
P0_exit_turb_low = 82.67 * 10**3           #Note: Check values
deltah0_turb_low = get_stagnation_enthalpy_change(Cp, T0_entry_turb_low, T0_exit_turb_low)
deltaP0_turb_low = get_stagnation_pressure_change(deltah0_turb_low, P_entry_turb_low)














#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Low Pressure Compressor Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

fig3D = plt.figure()
ax3D = fig3D.add_subplot(1,1,1, projection='3d')
ax3D.set_title('Low Pressure Compressor')

#Defining lists that will be used to graph 
omega_comp_low_list = []
hub_r_comp_low_list = []


n_comp_low_List = []
psi_comp_low_List = []
phi_comp_low_List = []


for omega in np.arange(500, 1000, 100):
    for hub_r in np.arange(0.7, 0.95, 0.05):
        
        omega_comp_low_list.append(omega)
        hub_r_comp_low_list.append(hub_r)
        
        
        #Calculating Compressor Values
        results_compressor_low = calculate_compressor(omega, hub_r, pressure_ratio_comp_low, P_entry_comp_low, P0_entry_comp_low, P0_exit_comp_low, T0_entry_comp_low, T0_exit_comp_low, pressure_ratio_comp_max, mass_flow, polyentropic_efficiency, Cp, gamma, R)
        
        n_comp_low_List.append(results_compressor_low['n_stages'])
        psi_comp_low_List.append(results_compressor_low['psiMean'])
        phi_comp_low_List.append(results_compressor_low['phi'])


        #Plotting
        plot_results(results_compressor_low, 'Low Pressure Compressor', fig, axs)
        

mapable = ax3D.scatter(omega_comp_low_list, hub_r_comp_low_list, phi_comp_low_List , c=psi_comp_low_List)
cbar = fig3D.colorbar(mapable, orientation='vertical')
        
ax3D.set_xlabel('omega')
ax3D.set_ylabel('hub_to_tip_ratio')
ax3D.set_zlabel('phi')
cbar.set_label('psi')

plt.show()   
                







#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#High Pressure Compressor Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

fig3D = plt.figure()
ax3D = fig3D.add_subplot(1,1,1, projection='3d')
ax3D.set_title('Low Pressure Compressor')

#Defining lists that will be used to graph 
omega_comp_high_list = []
hub_r_comp_high_list = []


n_comp_high_List = []
psi_comp_high_List = []
phi_comp_high_List = []



for omega in np.arange(500, 1000, 100):
    for hub_r in np.arange(0.7, 0.95, 0.05):
  
        omega_comp_high_list.append(omega)
        hub_r_comp_high_list.append(hub_r)
        
        
        #Calculating Compressor Values
        results_compressor_high = calculate_compressor(omega, hub_r, pressure_ratio_comp_high, P_entry_comp_high, P0_entry_comp_high, P0_exit_comp_high, T0_entry_comp_high, T0_exit_comp_high, pressure_ratio_comp_max, mass_flow, polyentropic_efficiency, Cp, gamma, R)
        
        n_comp_high_List.append(results_compressor_high['n_stages'])
        psi_comp_high_List.append(results_compressor_high['psiMean'])
        phi_comp_high_List.append(results_compressor_high['phi'])
        
        #Plotting
        plot_results(results_compressor_high, 'High Pressure Compressor', fig, axs)

      
mapable = ax3D.scatter(omega_comp_high_list, hub_r_comp_high_list, phi_comp_high_List , c=psi_comp_high_List)
cbar = fig3D.colorbar(mapable, orientation='vertical')
        
ax3D.set_xlabel('omega')
ax3D.set_ylabel('hub_to_tip_ratio')
ax3D.set_zlabel('phi')
cbar.set_label('psi')

plt.show()     
                















#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#High Pressure Turbine Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

fig3D = plt.figure()
ax3D = fig3D.add_subplot(1,1,1, projection='3d')
ax3D.set_title('High Pressure Turbine')

#Defining lists that will be used to graph 
omega_turb_high_list = []
hub_r_turb_high_list = []


n_turb_high_List = []
psi_turb_high_List = []
phi_turb_high_List = []



for omega in np.arange(500, 1000, 100):
    for hub_r in np.arange(0.7, 0.95, 0.05):
  
        omega_turb_high_list.append(omega)
        hub_r_turb_high_list.append(hub_r)
        
        
        #Calculating Compressor Values
        results_turbine_high = calculate_turbine(omega, hub_r, pressure_ratio_turb_high, P_entry_turb_high, P0_entry_turb_high, P0_exit_turb_high, T0_entry_turb_high, T0_exit_turb_high, pressure_ratio_turb_max, mass_flow, polyentropic_efficiency, Cp, gamma, R)
       
        
        n_turb_high_List.append(results_turbine_high['n_stages'])
        psi_turb_high_List.append(results_turbine_high['psiMean'])
        phi_turb_high_List.append(results_turbine_high['phi'])
        
        #Plotting
        plot_results(results_turbine_high, 'High Pressure Turbine', fig, axs)

      
mapable = ax3D.scatter(omega_turb_high_list, hub_r_turb_high_list, phi_turb_high_List , c=psi_turb_high_List)
cbar = fig3D.colorbar(mapable, orientation='vertical')
        
ax3D.set_xlabel('omega')
ax3D.set_ylabel('hub_to_tip_ratio')
ax3D.set_zlabel('phi')
cbar.set_label('psi')

plt.show()     
                










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Low Pressure Turbine Calcuations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Define Axis to plot on
fig, axs = plt.subplots(2, 3)

fig3D = plt.figure()
ax3D = fig3D.add_subplot(1,1,1, projection='3d')
ax3D.set_title('Low Pressure Turbine')

#Defining lists that will be used to graph 
omega_turb_low_list = []
hub_r_turb_low_list = []


n_turb_low_List = []
psi_turb_low_List = []
phi_turb_low_List = []



for omega in np.arange(500, 1000, 100):
    for hub_r in np.arange(0.7, 0.95, 0.05):
  
        omega_turb_low_list.append(omega)
        hub_r_turb_low_list.append(hub_r)
        
        
        #Calculating Compressor Values
        results_turbine_low = calculate_turbine(omega, hub_r, pressure_ratio_turb_low, P_entry_turb_low, P0_entry_turb_low, P0_exit_turb_low, T0_entry_turb_low, T0_exit_turb_low, pressure_ratio_turb_max, mass_flow, polyentropic_efficiency, Cp, gamma, R)
       
        
        n_turb_low_List.append(results_turbine_low['n_stages'])
        psi_turb_low_List.append(results_turbine_low['psiMean'])
        phi_turb_low_List.append(results_turbine_low['phi'])
        
        #Plotting
        plot_results(results_turbine_low, 'Low Pressure Turbine', fig, axs)

      
mapable = ax3D.scatter(omega_turb_low_list, hub_r_turb_low_list, phi_turb_low_List , c=psi_turb_low_List)
cbar = fig3D.colorbar(mapable, orientation='vertical')
        
ax3D.set_xlabel('omega')
ax3D.set_ylabel('hub_to_tip_ratio')
ax3D.set_zlabel('phi')
cbar.set_label('psi')

plt.show()     
                


































              
    

































