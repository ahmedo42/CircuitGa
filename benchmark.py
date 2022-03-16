import numpy as np
import pickle
from deap import algorithms, base, creator, tools
from custom import eaSimple
from itertools import product
from collections import OrderedDict
from CircuitGa.eval_engines.ngspice.TwoStageClass import *
import random
from blackbox import BlackBox
experiments_grid = {
    "cxpb" : np.arange(0.5,1.0,0.1).tolist(),
    "n_gen": np.arange(10,110,10).tolist(),
    "pop_size": np.arange(100,1100,100).tolist(),
    "cx_method": [tools.cxOnePoint,tools.cxTwoPoint],
}

def load_valid_specs():
    with open("valid_specs", 'rb') as f:
        specs = pickle.load(f)
        
    specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
    return specs

CIR_YAML = "CircuitGa/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
box = BlackBox(sim_env, CIR_YAML)
toolbox.register("generate", box.generate_random_params)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt,indpb=0.5,low=(0),up=(98))
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", box.simulate)


def evaluate():
    specs = load_valid_specs()
    for i in range(5):
        designs_met = 0
        random.seed(random.randrange(int(1e6)))
        performances = []
        for j , (gain,ibias,phm,ugbw) in enumerate(zip(specs["gain_min"],specs["ibias_max"],specs["phm_min"],specs["ugbw_min"])):
            target_specs = [gain,ibias,phm,ugbw]
            setattr(box, "target_specs", target_specs)
            pop = toolbox.population(n=100)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            pop, log = eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
            hof_performance = box.simulate(hof[0],result="cost")[0]
            if hof_performance > 0:
                designs_met += 1
            print(f"total achieved designs : {designs_met}/1000, currently at {j+1} design")
        perfomrances.append(designs_met)
        print(f"total achieved designs : {designs_met}/1000")
    print(performances)

"""
def run_benchmark():
    cxpbs , n_gens , pops , cx_methods = experiments_grid.values()
    experiments = product(cxpbs , n_gens , pops , cx_methods)
    valid_specs = load_valid_specs()
    n_specs = 1000
    for exp in experiments:
        print(exp)
        for i in range(0,n_specs):
            cxpb, n_gen, pop, cx_method = list(exp)
"""
            
if __name__ == "__main__":
    evaluate()