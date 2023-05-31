import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage import data
import numpyro
import numpyro.distributions as dist
import jax.random as random
import einops
import jax
import jax.numpy as jnp
from simple_seg_utils import *
from simple_seg_get_edges import *
from GNN_model import *
import jraph

from matplotlib.pylab import *
from jax import  numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
# import monai_einops
import einops
import optax
from flax.training import train_state  # Useful dataclass to keep train state
import h5py
import jax
from ml_collections import config_dict
from jax.config import config
from skimage.segmentation import mark_boundaries
import cv2
import functools
import flax.jax_utils as jax_utils
from jax_smi import initialise_tracking
import ml_collections
import time
import more_itertools
import toolz
from subprocess import Popen
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint
from datetime import datetime
from flax.training import orbax_utils
from flax.core.frozen_dict import freeze
import flax
from  functools import partial
 
def work_on_single_area(curr_id,mask_curr,image,is_label,epsilon):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    masked_image= jnp.multiply(image,filtered_mask)
    if(is_label):
       summ=jnp.sum(filtered_mask.flatten())
       meann=jnp.sum(masked_image.flatten())/(summ+epsilon)
       meann=harder_diff_round(meann)
       return meann
    return masked_image

v_work_on_single_area=jax.vmap(work_on_single_area,in_axes=(0,0,0,None,None))
v_v_work_on_single_area=jax.vmap(v_work_on_single_area,in_axes=(0,0,0,None,None))


def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
    tokens= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in,is_label,epsilon)
    if(is_label):
       tokens=einops.rearrange(tokens,'b a->b a')
       return tokens
    tokens=einops.rearrange(tokens,'b f x y c->(b f) x y c')
    return tokens

def iter_over_all_masks(masks,curr_image,shape_reshape_cfgs,shape_reshape_cfgs_old,initial_masks,is_label,epsilon):
    tokens_a = iter_over_masks(shape_reshape_cfgs,0,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon)
    tokens_b = iter_over_masks(shape_reshape_cfgs,1,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon)
    tokens_c = iter_over_masks(shape_reshape_cfgs,2,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon)
    tokens_d = iter_over_masks(shape_reshape_cfgs,3,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon)
    tokens= jnp.concatenate([tokens_a,tokens_b,tokens_c,tokens_d ])
    return tokens



class Simple_graph_net(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    def setup(self):
        cfg=self.cfg
        initial_masks= jnp.stack([
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
                ],axis=0)
        initial_masks=jnp.sum(initial_masks,axis=0)   
        initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
        self.initial_masks=initial_masks
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
        self.shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)
        edge_pairs=get_sorce_targets(cfg.orig_grid_shape)
        self.senders=edge_pairs[:,0]
        self.receivers=edge_pairs[:,1]
        self.n_node = initial_masks.shape[1]*initial_masks.shape[2]
        self.n_edge = self.senders.shape[0]
        self.gnn=GraphNetwork(mlp_features=[5,5,5], latent_size=10)

    
    @nn.compact
    def __call__(self, curr_image: jnp.ndarray,curr_label: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        tokens=iter_over_all_masks(masks,curr_image,self.shape_reshape_cfgs,self.shape_reshape_cfgs_old,self.initial_masks,False,self.cfg.epsilon)
        
        # print(f"ttttttttttt tokens {tokens.shape}")
        tokens=Conv_trio(self.cfg,40)(tokens)
        tokens=Conv_trio(self.cfg,40)(tokens)
        tokens= einops.rearrange(tokens,'f x y c-> f (x y c)')
        tokens= nn.Dense(features=6)(tokens)
        tokens= nn.relu(tokens)
        # Optionally you can add `global` information, such as a graph label.
        global_context = jnp.array([[1]]) # Same feature dims as nodes and edges.
        graph = jraph.GraphsTuple(
            nodes=tokens,
            edges=None,
            senders=self.senders,
            receivers=self.receivers,
            n_node=self.n_node,
            n_edge=self.n_edge,
            globals=global_context)
        new_graph=self.gnn(graph)
        tokens=new_graph.nodes

        tokens=nn.Dense(features=2)(tokens)

        # tokens= nn.softmax(tokens)
        label_svs=iter_over_all_masks(masks,curr_label,self.shape_reshape_cfgs,self.shape_reshape_cfgs_old,self.initial_masks,True,self.cfg.epsilon)
        label_svs= einops.rearrange(label_svs,'f a -> (f a)')

        #basically one hot encoding
        label_svs = jnp.stack([(1-label_svs),label_svs],axis=-1)
        losss=optax.softmax_cross_entropy(tokens, label_svs)
        return losss





