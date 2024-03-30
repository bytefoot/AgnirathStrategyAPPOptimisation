import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from config import N, T, DT, EPOCHS
from constants import CarData
from profiles import (
    calc_acceleration_profile,
    calc_battery_profile,
    calc_distance_profile,
    calc_total_distance
)
import constraints as Constraints
from solar import solar_irradiance_profile

INIT_VELOCITY = tf.constant([0.0])
# Initialize optimizer (Adam optimizer)
optimizer = tf.keras.optimizers.Adam()

# ----------------------------------------------------------------------------------------------------------
# Define optimization step function
@tf.function
def optimization_step(vprof, solar_irradiance_profile):
    with tf.GradientTape() as tape:
        velocity_profile = tf.concat([INIT_VELOCITY, vprof], axis=0)

        objective = -calc_total_distance(velocity_profile)
        # d_prof = calc_distance_profile(velocity_profile)
        acceleration_profile = calc_acceleration_profile(velocity_profile)
        battery_profile = calc_battery_profile(CarData.max_battery_capacity, velocity_profile, acceleration_profile, solar_irradiance_profile)

        constraints = Constraints.battery(battery_profile) + Constraints.acceleration(acceleration_profile) + Constraints.velocity(velocity_profile)
        loss = objective + constraints

    gradients = tape.gradient(loss, vprof)
    optimizer.apply_gradients([(gradients, vprof)])
    return loss, velocity_profile

# -----------------------------------------------------------------------------------------------------------
# Perfomring optimisation

initial_velocity_profile = tf.Variable(tf.ones(N-1))  # Placeholder initial guess

# Perform optimization loop
for i in range(EPOCHS+1):  # Example: 1000 optimization steps
    loss, optimized_velocity_profile = optimization_step(initial_velocity_profile, solar_irradiance_profile)
    if i % 1000 == 0:
        print(f"Step {i}, Loss: {loss.numpy()}")

# -----------------------------------------------------------------------------------------------------------
# Print results
print("Optimized Velocity Profile:", optimized_velocity_profile)
print("Total Distance:", calc_total_distance(optimized_velocity_profile).numpy())
print("Total Energy Consumption:", calc_battery_profile(CarData.max_battery_capacity, optimized_velocity_profile, calc_acceleration_profile(optimized_velocity_profile), solar_irradiance_profile))

distance_profile = calc_distance_profile(optimized_velocity_profile)
acceleration_profile = calc_acceleration_profile(optimized_velocity_profile)
battery_profile = calc_battery_profile(CarData.max_battery_capacity, optimized_velocity_profile, acceleration_profile, solar_irradiance_profile)

print(distance_profile, acceleration_profile, battery_profile, sep="\n")

# -----------------------------------------------------------------------------------------------------------
# Visualise profiles

plt.plot(np.arange(len(optimized_velocity_profile))*DT*100/T, optimized_velocity_profile)
plt.show()
plt.plot(np.arange(len(battery_profile))*DT, battery_profile)
plt.show()

plt.plot(np.arange(len(acceleration_profile))*DT, acceleration_profile)
plt.show()
plt.plot(np.arange(len(distance_profile))*DT, distance_profile)
plt.show()
plt.plot(np.arange(len(solar_irradiance_profile))*DT, solar_irradiance_profile)
plt.show()