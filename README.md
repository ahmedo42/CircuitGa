# CircuitGa: Automating Analog Circuit Design with Genetic Algorithms

Code for evaluating genetic algorithms on automating sizing of analog circuit design.

To generate a pickle file containing specs, use the script in this repo  


## Setup
Install Dependencies

```
pip install -r requirements.txt
```

Install Ngspice for simulation, On Ubuntu/Colab/Kaggle
```
sudo apt-get install -y ngspice
```

## Evaluation

Before running the algorithm, the circuit netlist must be modified in order to point to the right library files in your directory. To do this, run the following command:
```
python interface/eval_engines/ngspice/ngspice_inputs/correct_inputs.py 
```

Configure the algorithm from `custom.py` and the hyperparams from `benchmark.py` and then run:

```
python benchmark.py 
```
