# AgnirathStrategyAPPOptimisation

## Overview
A base optimisation model, to find optimum strategy for the Agnirath solar race car. A rudimentary is made for the Strategy Module application. The model utilizes TensorFlow, NumPy, and Matplotlib for implementation and visualization purposes. 

## Features
- Optimization of velocity profiles to enhance performance in various applications.
- Written in Python, making it accessible and easy to integrate into existing projects.
- Utilizes TensorFlow for efficient optimization algorithms.
- Visualization of optimization results using Matplotlib.

## Requirements
You can install them using the provided `requirements.txt` file:
```
pip install -r requirements.txt
```

## Usage
1. Clone this repository to your local machine.
2. Navigate to the code root directory.
3. Install the required dependencies as detailed previously
4. Run the optimization model by executing `model.py`.
   ```
   python model.py
   ```
   THe code will first run model to figure out optimisation and then display the optimal strategies in a window (using matplotlib)

## Analysis
The profile seem to vary on varying maximum battery charge at begining, as it tends to become a limit factor. Hence I've changed the batttery charge to obtain different strategies, to just play arround and check the rigorousness of the model.

### With 0.001J
![0.001J](/assets/0.001.png)

This graph showcases the model very well. First, the model accelertes for a short time to gain afinite velocity and then it maintains the constatn velocity of a short period while gaining power. Then once it attains a bit more power, it again does the same and goes to a higher velocity. Eventually it decides on an optimum velocity to maintain to maximise distance travelled.

### With 0.01J
![0.01J](/assets/0.01.png)

### With 5.1J
![5.1J](/assets/5.1.png)
