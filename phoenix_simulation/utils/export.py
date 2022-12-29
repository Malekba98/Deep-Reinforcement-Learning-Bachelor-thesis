"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import os
import os
import json
import pandas
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import phoenix_simulation


def get_file_contents(
        file_path: str,
        skip_header: bool = False
    ):
    """Open the file with given path and return Python object."""
    assert os.path.isfile(file_path), 'No file exists at: {}'.format(file_path)
    if file_path.endswith('.json'):  # return dict
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    elif file_path.endswith('.csv'):
        if skip_header:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        else:
            data = np.loadtxt(file_path, delimiter=",")
        if len(data.shape) == 2:  # use pandas for tables..
            data = pandas.read_csv(file_path)
    else:
        raise NotImplementedError
    return data


def dump_json(
        activation: str,
        scaling_parameters: np.ndarray,
        neural_network: torch.nn.Module,
        file_name_path: str
) -> None:
    """Dump neural network as JSON to disk.

    Uses the format defined by Matthias Kissel and Sven Gronauer.

    Parameters
    ----------
    activation
    scaling_parameters
    neural_network
    file_name_path

    Returns
    -------
    None
    """
    data = dict()
    # === Create Check Sum:
    # Use a vector filled with ones to validate correct output of NN on hardware
    with torch.no_grad():
        out = neural_network(torch.ones(scaling_parameters.shape[1]))
    # TODO: uncomment checksum line
    # data["check_sum"] = str(out.numpy().sum())
    data["scaling_parameters"] = scaling_parameters.tolist()
    # Write the activation function
    data["activation"] = activation

    data_entry_i = 0
    for layer_i, layer in enumerate(neural_network):
        if isinstance(layer, nn.Linear):
            data[str(data_entry_i)] = dict()
            data[str(data_entry_i)]["type"] = "standard"
            # get the weights
            weights = layer.weight.detach().cpu().numpy().tolist()
            data[str(data_entry_i)]["weights"] = weights
            # get the biases
            biases = layer.bias.detach().cpu().numpy().tolist()
            data[str(data_entry_i)]["biases"] = biases
            data_entry_i += 1
    # print(data)
    print('-' * 55)
    print(neural_network)
    # raise
    save_file_name_path = os.path.join(file_name_path, "model.json")
    with open(save_file_name_path, 'w') as outfile:
        json.dump(data, outfile)
    print('-' * 55)
    print(f'JSON saved at: {save_file_name_path}')


def convert_actor_critic_to_json(
        actor_critic: torch.nn.Module,
        file_name_path
):
    """Save PyTorch Module as json to disk."""
    # Write the headers
    scaling_parameters = np.empty((2, 16))
    scaling_parameters[0] = actor_critic.obs_oms.mean.numpy()
    scaling_parameters[1] = actor_critic.obs_oms.std.numpy()
    # print(scaling_parameters)
    # raise NotImplementedError
    dump_json(
        activation=actor_critic.ac_kwargs['pi']['activation'],
        scaling_parameters=scaling_parameters,
        neural_network=actor_critic.pi.net,
        file_name_path=file_name_path
    )


def load_actor_critic_and_env_from_disk(
        file_name_path: str
) -> tuple:
    """Loads ac module from disk. (@Sven).

    Parameters
    ----------
    file_name_path

    Returns
    -------
    tuple
        holding (actor_critic, env)
    """
    try:
        import research
        from research.algs import core
    except ImportError:
        raise ValueError('You have no access to Sven´s research repository :-)')

    config_file_path = os.path.join(file_name_path, 'config.json')
    conf = get_file_contents(config_file_path)
    print('Loaded config file:')
    print(conf)
    env_id = conf.get('env_id')
    env = gym.make(env_id)
    alg = conf.get('alg', 'ppo')
    try:
        ac = core.ActorCriticWithCosts(
            actor_type=conf['actor'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=conf['use_standardized_obs'],
            use_scaled_rewards=conf['use_reward_scaling'],
            use_shared_weights=False,
            ac_kwargs=conf['ac_kwargs']
        )
        model_path = os.path.join(file_name_path, 'torch_save', 'model.pt')
        ac.load_state_dict(torch.load(model_path))
    except RuntimeError:
        ac = core.ActorCritic(
            actor_type=conf['actor'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=conf['use_standardized_obs'],
            use_scaled_rewards=conf['use_reward_scaling'],
            use_shared_weights=conf['use_shared_weights'],
            ac_kwargs=conf['ac_kwargs']
        )
    model_path = os.path.join(file_name_path, 'torch_save', 'model.pt')
    ac.load_state_dict(torch.load(model_path))
    print(f'Successfully loaded model from: {model_path}')
    return ac, env
