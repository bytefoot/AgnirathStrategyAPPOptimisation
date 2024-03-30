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
The profile seem to vary on varying maximum battery charge at begining, as it tends to be coem a limit factor. Hence I've changed the batttery charge to obtain different strategies, to just play arround and check the rigorousness of the model.

#### With 0.0001J
![0.0001J](/assets/0.0001.png)

#### With 0.1J
![0.1J](/assets/0.1.png)

#### With 1J
![1J](/assets/1.png)
