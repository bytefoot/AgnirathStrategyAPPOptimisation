import tensorflow as tf
import numpy as np
from itertools import pairwise

from config import T, DT
from constants import CarData

# Solar data measured at 9:00, 10:00, ... 19:00
hourly_inclination_sample = [0, 0, 30, 45, 30, 15, 0, 0, 0, 30, 0]

# ----------------------------------------------------------------------------------------------------------
# Trying to interpolate and augment solar data for the time in between sampling

# Calculate number of intervals between each pair of points
num_intervals = T // (len(hourly_inclination_sample) - 1) // DT

# actual interpolation
_inclination_profile_degrees = tf.constant([
    augmented_data
    for start_v, end_v in pairwise(hourly_inclination_sample)
    for augmented_data in np.linspace(start_v, end_v, num_intervals)
], dtype=tf.float32)

# Convertion of units
inclination_profile = _inclination_profile_degrees * np.pi / 180.0

inclination_diff_profile = tf.constant(
    np.concatenate((
        np.array([inclination_profile[0]]),
        np.array([
            (0 if inclination_profile[i] == inclination_profile[i-1] else (1 if inclination_profile[i] > inclination_profile[i-1] else -1))
            for i in range(1, len(inclination_profile))
        ])
    ))
, dtype=tf.float32)

net_resistive_incline_coefficient = CarData.fric_coeff*np.cos(inclination_profile) + inclination_diff_profile * np.sin(inclination_profile)
# net_resistive_incline_coefficient = CarData.fric_coeff * np.ones_like(inclination_profile)
