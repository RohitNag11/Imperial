import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def draw_engine(engine):
    fig, ax = plt.subplots()
    ax.set_title('Engine Geometry')
    ax.set_ylabel('Radius (m)')
    ax.set_xlabel('Stages')
    ax.set_ylim(-engine.diameter/2 - 0.5,
                engine.diameter/2 + 0.5)
    draw_engine_casing(ax, engine.diameter)
    draw_fan(ax, engine.fan.tip_diameter,
             engine.fan.inner_fan_tip_diameter, engine.fan.hub_diameter, 0, 'Fan')
    draw_fan(ax, -engine.fan.tip_diameter,
             -engine.fan.inner_fan_tip_diameter, -engine.fan.hub_diameter, 0)
    component_gap = 5
    start_pos = 6
    turb_components = [engine.lpc, engine.hpc, engine.hpt, engine.lpt]
    names = ['LPC', 'HPC', 'HPT', 'LPT']
    for i, turb_comp in enumerate(turb_components):
        draw_turb_comp(ax, turb_comp.hub_diameters,
                       turb_comp.tip_diameters, start_pos, names[i])
        draw_turb_comp(ax, -turb_comp.hub_diameters,
                       -turb_comp.tip_diameters, start_pos)
        start_pos += 2 * turb_comp.no_of_stages + component_gap
    plt.show()


def draw_turb_comp(ax, hub_diameters, tip_diameters, axial_start_position, name=None):
    hub_diameters = np.repeat(hub_diameters, 2)
    tip_diameters = np.repeat(tip_diameters, 2)
    mean_radius = (hub_diameters + tip_diameters) / 4
    n_stages_double = len(hub_diameters)
    z = np.linspace(axial_start_position,
                    axial_start_position + n_stages_double, n_stages_double) / 2

    stage_polygons_points = [[(z[i], tip_diameters[i] / 2),
                              (z[i], hub_diameters[i] / 2),
                              (z[i+1], hub_diameters[i+1] / 2),
                              (z[i+1], tip_diameters[i+1] / 2)]
                             for i in range(0, n_stages_double, 2)]

    [ax.add_patch(Polygon(points, alpha=0.5, color='b'))
     for points in stage_polygons_points]
    [ax.plot([z[i], z[i]], [hub_diameters[i] / 2, tip_diameters[i] / 2], 'k')
     for i in range(n_stages_double)]
    ax.plot(z, hub_diameters / 2, 'k')
    ax.plot(z, tip_diameters / 2, 'k')
    ax.plot(z, mean_radius, 'r', ls='-.')
    ax.text(np.mean(z[:-1]), mean_radius[-1]+0.1, name, color='r')


def draw_fan(ax, fan_tip, inner_fan_tip, fan_hub, axial_start_position, name=None):
    hub_diameters = np.repeat(fan_hub, 2)
    tip_diameters = np.repeat(fan_tip, 2)
    inner_fan_tip_diameters = np.repeat(inner_fan_tip, 2)
    inner_fan_mean_radius = (inner_fan_tip_diameters + hub_diameters) / 4
    n_stages_double = len(hub_diameters)
    z = np.linspace(axial_start_position,
                    axial_start_position + n_stages_double, n_stages_double) / 2
    inner_fan_polygons_points = [(z[0], inner_fan_tip_diameters[0] / 2),
                                 (z[0], hub_diameters[0] / 2),
                                 (z[1], hub_diameters[1] / 2),
                                 (z[1], inner_fan_tip_diameters[1] / 2)]
    outer_fan_polygons_points = [(z[0], tip_diameters[0] / 2),
                                 (z[0], inner_fan_tip_diameters[0] / 2),
                                 (z[1], inner_fan_tip_diameters[1] / 2),
                                 (z[1], tip_diameters[1] / 2)]
    ax.add_patch(Polygon(inner_fan_polygons_points, alpha=0.5, color='b'))
    ax.add_patch(Polygon(outer_fan_polygons_points, alpha=0.2, color='b'))
    ax.plot(z, hub_diameters / 2, 'k')
    ax.plot(z, tip_diameters / 2, 'k')
    [ax.plot([z[i], z[i]], [hub_diameters[i] / 2, tip_diameters[i] / 2], 'k')
     for i in range(n_stages_double)]
    ax.plot(z, inner_fan_tip_diameters / 2, 'k')
    ax.plot(z, inner_fan_mean_radius, 'r', ls='-.')
    ax.text(z[0], tip_diameters[0] / 2 + 0.1, name, c='r')


def draw_engine_casing(ax, diameter):
    ax.axhline(0, color='black', linestyle='-.')
    ax.axhline(diameter / 2, color='black', linestyle='-')
    ax.axhline(-diameter / 2, color='black', linestyle='-')
