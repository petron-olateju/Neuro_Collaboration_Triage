import os
from typing import List

import yaml
import numpy as np
import torch


class Experiment:

  def __init__(self, name, dir, description, baseline=None):
    self.name = name
    self.dir = f'{dir}/{name}'
    self.description = description
    self.parameters = dict()
    self.baseline = baseline

  def update_param(self, parameter: 'Parameter'):
    var_name = parameter.get_var_name()
    parameter_class = parameter.get_parameter_class()
    value = parameter.get_value()

    assert isinstance(
        value,
        (
            int, float, str, dict, list,
            np.ndarray, torch.tensor, None
        )
    )

    if (parameter_class is None) or (parameter_class.lower() == 'global'):
      self.parameters[var_name] = value
      return

    if parameter_class not in self.parameters:
      self.parameters[parameter_class] = dict()
    self.parameters[parameter_class][var_name] = value

  def save(self):
    experiment_dict = {
        'name': self.name,
        'baseline': self.baseline,
        'description': self.description,
        'parameters': self.parameters,
    }

    os.makedirs(self.dir, exist_ok=True)

    with open(f'{self.dir}/training_parameters.yaml', "w") as f:
      yaml.dump(experiment_dict, f, default_flow_style=False)

    print(f"Model saved to {self.dir}")

  def add_params(self, parameters: List['Parameter']):

    for parameter in parameters:
      self.update_param(parameter)


class Parameter:

  def __init__(self, value, var_name, parameter_class):
    self.__var_name = var_name
    self.__value = value
    self.__parameter_class = parameter_class

  def get_var_name(self):
    return self.__var_name

  def get_parameter_class(self):
    return self.__parameter_class

  def get_value(self):
    return self.__value

  def set_value(self, value):
    self.__value = value

  def set_var_name(self, var_name):
    self.__var_name = var_name

  def set_parameter_class(self, parameter_class):
    self.__parameter_class = parameter_class