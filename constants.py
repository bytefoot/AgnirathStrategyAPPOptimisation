class Physics:
    g = 9.8

class UnitScale:
    kWh = 1000*3600

class CarData:
    mass = 280  # kgs
    aero_drag = 0.12 * 1  # SI
    fric_coeff = 0.045  # SI

    max_battery_capacity = 0.00009 * UnitScale.kWh

    motor_efficiency = 0.98

    max_acceleration = 1  # m/s^2
    max_velocity = 30  # m/s

    panel_area = 4  # m^2
