import tensorflow as tf
import numpy as np
from itertools import pairwise

from config import T, DT

# Solar data measured at 9:00, 10:00, ... 19:00
hourly_solar_irradiance_sample = [164, 403, 637, 851, 986, 1046, 1029, 936, 776, 569, 421]

# ----------------------------------------------------------------------------------------------------------
# Trying to interpolate and augment solar data for the time in between sampling

# Calculate number of intervals between each pair of points
num_intervals = T // (len(hourly_solar_irradiance_sample) - 1) // DT

# actual interpolation
solar_irradiance_profile = tf.constant([
    augmented_data
    for start_v, end_v in pairwise(hourly_solar_irradiance_sample)
    for augmented_data in np.linspace(start_v, end_v, num_intervals)
], dtype=tf.float32)
