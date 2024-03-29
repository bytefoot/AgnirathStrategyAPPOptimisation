import tensorflow as tf

from config import DT
from constants import CarData, Physics

def calc_distance_profile(velocity_profile):
    return tf.cumsum((velocity_profile[:-1] + velocity_profile[1:]) / 2 * DT)

def calc_total_distance(velocity_profile):
    return tf.reduce_sum((velocity_profile[:-1] + velocity_profile[1:]) / 2 * DT)

def calc_acceleration_profile(velocity_profile):
    return (velocity_profile[1:] - velocity_profile[:-1]) / DT

def power_equation(accleration, velocity):
    return CarData.mass*accleration*velocity + CarData.aero_drag*(velocity**3) + CarData.fric_coeff*CarData.mass*Physics.g*velocity

def calc_power_profile(_velocity_profile, acceleration_profile):
    velocity_profile = (_velocity_profile[:-1] + _velocity_profile[1:]) / 2

    return power_equation(acceleration_profile, velocity_profile) / CarData.motor_efficiency

def calc_battery_profile(max_cap, v_profile, a_prof, s_prof):
    pp = calc_power_profile(v_profile, a_prof)
    return max_cap + tf.cumsum( - pp*DT + s_prof*CarData.panel_area)