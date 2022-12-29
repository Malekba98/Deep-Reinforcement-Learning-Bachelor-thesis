"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
r"""This is a function used by Sven to extract the policy networks from
his trained Actor Critic module and convert it to the ONNX file format.

"""

import argparse
import os
import torch
import phoenix_simulation
from phoenix_simulation.utils import export


def convert_to_onxx_file_format(file_name_path):
    """ This method is used by Sven to convert his trained actor critic
    checkpoints to the ONXX format.

    """
    ac, env = export.load_actor_critic_and_env_from_disk(file_name_path)

    # Actor network is of shape:
    # --------------------------
    # Sequential(
    #   (0): Linear(in_features=12, out_features=64, bias=True)
    #   (1): Tanh()
    #   (2): Linear(in_features=64, out_features=64, bias=True)
    #   (3): Tanh()
    #   (4): Linear(in_features=64, out_features=4, bias=True)
    #   (5): Identity()
    # )
    print(ac.pi.net)
    #
    dummy_input = torch.ones(*env.observation_space.shape)
    # print('dummy_input: ', dummy_input)
    model = ac.pi.net

    print('=' * 55)
    print('Test ac.pi.net....')
    for name, param in model.named_parameters():
        print(name)
    print('=' * 55)

    torch.save(model.state_dict(), os.path.join(file_name_path, "ActorMLP.pt"))

    save_file_name_path = os.path.join(file_name_path, "ActorMLP.onnx")
    torch.onnx.export(model, dummy_input, save_file_name_path,
                      verbose=True,
                      # opset_version=12,  # ONNX version to export the model to
                      export_params=True,  # store the trained parameters
                      do_constant_folding=False,  # necessary to preserve parameters names!
                      # input_names=["input"],
                      # output_names=["output"]
    )

    # Save observation standardization
    print('=' * 55)
    print('Save observation standardization...')
    model = ac.obs_oms
    save_file_name_path = os.path.join(file_name_path, "ObsStand.onnx")
    torch.onnx.export(model, dummy_input, save_file_name_path,
                      verbose=True,
                      # opset_version=12,  # ONNX version to export the model to
                      export_params=True,  # store the trained parameters
                      do_constant_folding=False,  # necessary to preserve parameters names!
                      # input_names=["input"],
                      # output_names=["output"]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Name path of the file to be converted.}')
    parser.add_argument('--output', type=str, default='json',
                        help='Choose output file format: [onnx, json].}')
    args = parser.parse_args()

    assert os.path.exists(args.ckpt)
    if args.output == 'onnx':
        # load a saved checkpoint file (.pt) from Sven's research repository
        # extract the actor network from the ActorCritic module
        # and save as .ONXX file to disk space
        convert_to_onxx_file_format(args.ckpt)
    elif args.output == 'json':
        # Convert PyTorch module as JSON file and save to disk.
        ac, env = export.load_actor_critic_and_env_from_disk(args.ckpt)
        export.convert_actor_critic_to_json(
            actor_critic=ac,
            file_name_path=args.ckpt
        )

    else:
        raise ValueError('Expecting json or onnx as file output.')
