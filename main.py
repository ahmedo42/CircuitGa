import random
import numpy as np
import argparse
from CircuitGa.eval_engines.ngspice.TwoStageClass import *
from CircuitGa.eval_engines.ngspice.csamp import *
from deap import algorithms, base, creator, tools
from deap.algorithms import eaSimple
from blackbox import *

parser = argparse.ArgumentParser()
parser.add_argument("--env",type=str,default="two_stage_opamp")
parser.add_argument("--seed",type=int,default=17)
parser.add_argument("--n_pop",type=int,default=100)
parser.add_argument("--n_gen",type=int,default=3)
args = parser.parse_args()

random.seed(args.seed)
CIR_YAML = f"CircuitGa/eval_engines/ngspice/ngspice_inputs/yaml_files/{args.env}.yaml"
if args.env == "two_stage_opamp":
    sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())
else:
    sim_env = CsAmpClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())

box = BlackBox(sim_env, CIR_YAML)


def main():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("generate", box.generate_random_params)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,indpb=0.1,low=(0),up=(98))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", box.simulate)
    
    pop = toolbox.population(n=args.n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, ngen=args.n_gen, stats=stats, halloffame=hof, verbose=True)
    best_individual = tools.selBest(pop, k=1)[0]
    print('best params',best_individual)
    print("target design",box.target_specs)
    print("best achieved design",box.simulate(best_individual,result="specs"))

if __name__=="__main__":
  main()
