import tensorflow as tf

from config import STRICT_PENALTY
from constants import CarData

def battery(battery_profile):
    # m = tf.reduce_min(battery_profile)
    # const1 = tf.maximum(0.0, STRICT_PENALTY * -m)
    # const2 = (m)**-10 if m >= 0 else 0.0
    # return const1 + const2
    upper_lim = tf.maximum(0.0, STRICT_PENALTY * tf.reduce_max(battery_profile - CarData.max_battery_capacity))
    lower_lim = tf.maximum(0.0, STRICT_PENALTY * -tf.reduce_min(battery_profile - 0.05*CarData.max_battery_capacity))
    return upper_lim + lower_lim

def velocity(velocity_profile):
    upper_lim = tf.maximum(0.0, STRICT_PENALTY*(tf.reduce_max(velocity_profile - CarData.max_velocity)))
    lower_lim = tf.maximum(0.0, -STRICT_PENALTY*(tf.reduce_min(velocity_profile)))
    return upper_lim + lower_lim

def acceleration(acceleration_profile):
    return tf.maximum(0.0, STRICT_PENALTY*tf.reduce_max(tf.abs(acceleration_profile) - CarData.max_acceleration))