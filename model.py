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

initial_velocity_profile = tf.Variable(tf.ones(N))  # Placeholder initial guess
print("optimising...\n")

# Perform optimization loop
for i in range(EPOCHS+1):  # Example: 1000 optimization steps
    loss, optimized_velocity_profile = optimization_step(initial_velocity_profile, solar_irradiance_profile)
    if i % 1000 == 0:
        print(f"Step {i}/{EPOCHS}, Loss: {loss.numpy()}")

print("="*50)
# -----------------------------------------------------------------------------------------------------------
# Print results
print("Optimized Velocity Profile:", optimized_velocity_profile)
print("Total Distance:", calc_total_distance(optimized_velocity_profile).numpy())
# print("Total Energy Consumption:", calc_battery_profile(CarData.max_battery_capacity, optimized_velocity_profile, calc_acceleration_profile(optimized_velocity_profile), solar_irradiance_profile))

distance_profile = calc_distance_profile(optimized_velocity_profile)
acceleration_profile = calc_acceleration_profile(optimized_velocity_profile)
battery_profile = calc_battery_profile(CarData.max_battery_capacity, optimized_velocity_profile, acceleration_profile, solar_irradiance_profile)

# print(distance_profile, acceleration_profile, battery_profile, sep="\n")
print("="*50)
# -----------------------------------------------------------------------------------------------------------
# Visualise profiles
print("Plotting profiles ...")
print("Please resize window to increase scale of graphs.")
print("Close the plot window to exit program.")

ax = plt.subplot(3, 1, 1)
plt.plot(np.arange(len(optimized_velocity_profile))*DT, optimized_velocity_profile)
plt.title("Velocity Profile")
plt.xticks(np.arange(T//N + 1)*N)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Velocity (m/s)")

ax = plt.subplot(3, 2, 3)
plt.plot(np.arange(len(battery_profile))*DT, battery_profile)
plt.title("Battery Profile")
ax.spines['left'].set_position(('data', 0))  # Set left spine position to x=0
ax.spines['bottom'].set_position(('data', 0))  # Set bottom spine position to y=0
plt.xticks(np.arange(T//N + 1)*N)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Energy (J)")


ax = plt.subplot(3, 2, 4)
plt.plot(np.arange(len(acceleration_profile))*DT, acceleration_profile)
plt.title("Acceleration Profile")
plt.xticks(np.arange(T//N + 1)*N)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Acceleration (m/s^2)")

ax = plt.subplot(3, 2, 5)
plt.plot(np.arange(len(distance_profile))*DT, distance_profile)
plt.title("Distance Profile")
plt.xticks(np.arange(T//N + 1)*N)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Distance (m)")

ax = plt.subplot(3, 2, 6)
plt.plot(np.arange(len(solar_irradiance_profile))*DT, solar_irradiance_profile)
plt.title("Solar Irradiance Profile")
plt.xticks(np.arange(T//N + 1)*N)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Irradiance (J/m^2)")

plt.subplots_adjust(hspace=1, wspace=1)  # Add padding between plots
plt.tight_layout()  # Adjusts subplot layout to prevent overlapping

wm = plt.get_current_fig_manager()
wm.canvas.manager.set_window_title("Profile Visualisation")
wm.canvas.manager.full_screen_toggle()
plt.show()

print('Exiting...')