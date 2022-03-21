import numpy as np
import pickle
from deap import algorithms, base, creator, tools
from custom import eaSimple
from itertools import product
from collections import OrderedDict
from CircuitGa.eval_engines.ngspice.TwoStageClass import *
import random
from blackbox import BlackBox
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cxpb",type=float,default=0.5)
parser.add_argument("--mutpb",type=float,default=0.5)
parser.add_argument("--mut_indpb",type=float,default=0.5)
parser.add_argument("--cx_indpb",type=float,default=0.5)
parser.add_argument("--pop",type=int,default=300)
parser.add_argument("--gen",type=int,default=50)
args = parser.parse_args()

def load_valid_specs():
    with open("valid_specs", 'rb') as f:
        specs = pickle.load(f)
        
    specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
    return specs




def evaluate(toolbox,box):
    specs = load_valid_specs()
    designs_met = 0
    random.seed(random.randrange(int(1e6)))
    performances = []
    n_evals = []
    for j , (gain,ibias,phm,ugbw) in enumerate(zip(specs["gain_min"],specs["ibias_max"],specs["phm_min"],specs["ugbw_min"])):
        if j == 100:
            break
        target_specs = [gain,ibias,phm,ugbw]
        setattr(box, "target_specs", target_specs)
        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=50, stats=stats, halloffame=hof, verbose=True)
        hof_performance = box.simulate(hof[0],result="cost")[0]
        if hof_performance > 0:
            designs_met += 1
        print(f"total achieved designs : {designs_met}/1000, currently at {j+1} ")
        n_evals.append(np.mean(log.select("nevals")))
        print("#"*10)
    print(np.mean(n_evals))
            
if __name__ == "__main__":
    CIR_YAML = "CircuitGa/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
    sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    box = BlackBox(sim_env, CIR_YAML)
    toolbox.register("generate", box.generate_random_params)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform,indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt,indpb=0.5,low=(0),up=(98))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", box.simulate)
    evaluate(toolbox,box)