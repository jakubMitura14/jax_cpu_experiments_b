# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example training script for training OGB molhiv with jax graph-nets & flax.

The ogbg-molhiv dataset is a molecular property prediction dataset.
It is adopted from the MoleculeNet [1]. All the molecules are pre-processed
using RDKit [2].

Each graph represents a molecule, where nodes are atoms, and edges are chemical
bonds. Input node features are 9-dimensional, containing atomic number and
chirality, as well as other additional atom features such as formal charge and
whether the atom is in the ring or not.

The goal is to predict whether a molecule inhibits HIV virus replication or not.
Performance is measured in ROC-AUC.

This script uses a GraphNet to learn the prediction task.

[1] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes,
Caleb Geniesse, Aneesh SPappu, Karl Leswing, and Vijay Pande.
Moleculenet: a benchmark for molecular machine learning.
Chemical Science, 9(2):513–530, 2018.

[2] Greg Landrum et al. RDKit: Open-source cheminformatics, 2006.

Example usage:

python3 train.py --data_path={DATA_PATH} --master_csv_path={MASTER_CSV_PATH} \
--save_dir={SAVE_DIR} --split_path={SPLIT_PATH}
"""


import functools
import logging
import pathlib
import pickle
from typing import Sequence
from absl import app
from absl import flags
from flax import linen as nn
import jax
import jax.numpy as jnp
import jraph
from jraph.ogb_examples import data_utils



# flags.DEFINE_integer('batch_size', 1, 'Number of graphs in batch.')
# flags.DEFINE_integer('num_training_steps', 10, 'Number of training steps.')
# flags.DEFINE_enum('mode', 'train', ['train', 'evaluate'], 'Train or evaluate.')
# FLAGS = flags.FLAGS

class ExplicitMLP(nn.Module):
  """A flax MLP."""
  features: Sequence[int]
  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
      x = lyr(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x



def make_mlp(features):
  @jraph.concatenated_args
  def update_fn(inputs):
    return ExplicitMLP(features)(inputs)
  return update_fn


class GraphNetwork(nn.Module):
  """A flax GraphNetwork."""
  mlp_features: Sequence[int]
  latent_size: int


  @nn.compact
  def __call__(self, graph):
    # Add a global parameter for graph classification.
    # embedder = jraph.GraphMapFeatures(
    #     embed_node_fn=make_embed_fn(self.latent_size),
    #     embed_edge_fn=make_embed_fn(self.latent_size),
    #     embed_global_fn=make_embed_fn(self.latent_size))
    net = jraph.GraphNetwork(
        update_node_fn=make_mlp(self.mlp_features),
        update_edge_fn=None,
        update_global_fn=None)  
        # update_edge_fn=lambda edges: edges,
        # update_global_fn=lambda globall: globall)  
    return net(graph)



