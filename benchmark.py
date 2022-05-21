import argparse
import pickle
import random
from collections import OrderedDict

import numpy as np
from deap import base, creator, tools

from blackbox import BlackBox
from custom import eaSimple
from interface.eval_engines.ngspice.TwoStageClass import *

parser = argparse.ArgumentParser()
parser.add_argument("--cxpb", type=float, default=0.5)
parser.add_argument("--mutpb", type=float, default=0.5)
parser.add_argument("--mut_indpb", type=float, default=0.5)
parser.add_argument("--cx_indpb", type=float, default=0.5)
parser.add_argument("--pop", type=int, default=300)
parser.add_argument("--ngen", type=int, default=50)
parser.add_argument("--env", type=str, default="two_stage_opamp")
args = parser.parse_args()


def load_valid_specs():
    with open("specs_valid_two_stage_opamp", "rb") as f:
        specs = pickle.load(f)

    specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
    return specs


def evaluate(toolbox, box):
    specs = load_valid_specs()
    n_specs = len(list(specs.values())[0])
    designs_met = 0
    random.seed(15)
    n_evals = []
    for i in range(n_specs):
        target_specs = [spec[i] for spec in specs.values()]
        setattr(box, "target_specs", target_specs)
        pop = toolbox.population(n=args.pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = eaSimple(
            pop,
            toolbox,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            ngen=args.ngen,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
        hof_performance = box.simulate(hof[0], result="cost")[0]
        if hof_performance > 0:
            designs_met += 1
        print(f"total achieved designs : {designs_met}/{i+1}")
        n_evals.append(np.sum(log.select("nevals")))
        print("#" * 10)
    print(f"Simulations per design: {np.mean(n_evals)}")


if __name__ == "__main__":
    CIR_YAML = (
        f"interface/eval_engines/ngspice/ngspice_inputs/yaml_files/{args.env}.yaml"
    )
    if args.env == "two_stage_opamp":
        sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())

    box = BlackBox(sim_env, CIR_YAML)
    param_upper_limit = tuple([len(param_vec) - 1 for param_vec in box.params])
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("generate", box.generate_random_params)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.generate
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=args.cx_indpb)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        indpb=args.mut_indpb,
        low=(0) * len(box.params_id),
        up=param_upper_limit,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", box.simulate)
    evaluate(toolbox, box)
