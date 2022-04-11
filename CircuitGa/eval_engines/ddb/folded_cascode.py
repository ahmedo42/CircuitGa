import os
import pickle
import random
from collections import OrderedDict

import gym
import numpy as np
import pandas as pd
from yaml_loader import OrderedDictYAMLLoader
from gym import spaces
from scipy import spatial 

class FoldedCascode:
    def __init__(self,df_path,yaml_path):
        self.df = pd.read_csv(df_path)
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        self.specs_id = sorted(list(yaml_data['target_specs'].keys()))
        self.params_id = sorted(list(params_dict.keys()))

    def _get_closest_spec(self,params):
        if not hasattr(self, "params_tree"):
            params_df = self.df[self.params_id]
            params_df = params_df.reindex(sorted(params_df.columns), axis=1)
            self.params_tree = spatial.KDTree(params_df.values) 
        
        idx = self.params_tree.query(params)[1]
        result = self.df.iloc[[idx]]
        result = result[self.specs_id]
        result = result.reindex(sorted(result.columns), axis=1)
        return result.values[0]

    def update(self,params):
        cur_specs = self._get_closest_spec(params)
        return cur_specs