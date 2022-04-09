import yaml
import random
import numpy as np
from yaml_loader import OrderedDictYAMLLoader
from collections import OrderedDict

class BlackBox:
    def __init__(self,sim_env,yaml_file,target_specs=None):
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        self.simulator = sim_env
        self._init_params(yaml_data['params'])
        self.target_specs = target_specs
        self.specs_id = sorted(list(yaml_data['target_specs'].keys()))

    def _init_params(self,params_dict):
        param_vals = list(params_dict.values())
        self.params = []
        self.params_id = list(params_dict.keys())
        for value in params_dict.values():
            param_vec = np.arange(value[0], value[1], value[2]).tolist()
            self.params.append(param_vec)

    def _normalize(self,spec):
        goal_spec = np.array([float(e) for e in self.target_specs])
        norm_spec = (spec-goal_spec)/(goal_spec+spec)
        return norm_spec

    def _calculate_cost(self,spec):
        rel_specs = self._normalize(np.array(spec))
        cost = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if self.specs_id[i] == 'ibias_max' or self.specs_id[i] == "IB":
                rel_spec *= -1.0
            if rel_spec < 0:
                cost += rel_spec

        cost = cost if cost < -0.02 else sum(rel_specs)
        return [cost]

    def simulate(self,design,result="cost"):
        params_idx = design
        new_params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        if hasattr(self.simulator,"df"):
            specs = self.simulator.update(new_params)
        else:
            param_val = [OrderedDict(list(zip(self.params_id,new_params)))]
            specs = OrderedDict(sorted(self.simulator.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
            specs = list(specs.values())
        return self._calculate_cost(specs) if result == "cost" else specs

    def generate_random_params(self):
        return  [random.randint(0, len(param_vec)-1) for param_vec in self.params]