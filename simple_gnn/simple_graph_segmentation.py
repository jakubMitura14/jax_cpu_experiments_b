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
 
def work_on_single_area(curr_id,mask_curr,image,is_label,epsilon,is_to_recreate):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    masked_image= jnp.multiply(image,filtered_mask)
    if(is_label):
       summ=jnp.sum(filtered_mask.flatten())
       meann=jnp.sum(masked_image.flatten())/(summ+epsilon)
       meann=harder_diff_round(meann)
       return meann
    if(is_to_recreate):
        return filtered_mask
    return masked_image

v_work_on_single_area=jax.vmap(work_on_single_area,in_axes=(0,0,0,None,None,None))
v_v_work_on_single_area=jax.vmap(v_work_on_single_area,in_axes=(0,0,0,None,None,None))


def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
    tokens= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in,is_label,epsilon,is_to_recreate)
    if(is_label):
       tokens=einops.rearrange(tokens,'b a->b a')
       return tokens
    tokens=einops.rearrange(tokens,'b f x y c->(b f) x y c')
    return tokens

def iter_over_all_masks(masks,curr_image,shape_reshape_cfgs,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate=False):
    tokens_a = iter_over_masks(shape_reshape_cfgs,0,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_b = iter_over_masks(shape_reshape_cfgs,1,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_c = iter_over_masks(shape_reshape_cfgs,2,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_d = iter_over_masks(shape_reshape_cfgs,3,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens= jnp.concatenate([tokens_a,tokens_b,tokens_c,tokens_d ])
    return tokens



def iter_over_all_masks_for_shape(masks,curr_image,shape_reshape_cfgs,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate=True):
    tokens_a = iter_over_masks(shape_reshape_cfgs,0,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_b = iter_over_masks(shape_reshape_cfgs,1,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_c = iter_over_masks(shape_reshape_cfgs,2,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    tokens_d = iter_over_masks(shape_reshape_cfgs,3,masks,curr_image,shape_reshape_cfgs_old,initial_masks,is_label,epsilon,is_to_recreate)
    return (tokens_a,tokens_b,tokens_c,tokens_d   ,tokens_a.shape, tokens_b.shape,tokens_c.shape,tokens_d.shape )



def get_orgig_token_shape(token,shape_reshape_cfg):
    to_reshape_back_x=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter_x
    to_reshape_back_y=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter_y
    token = einops.rearrange(token, ' f x y c-> 1 f x y c')
    return recreate_orig_shape(token,shape_reshape_cfg,to_reshape_back_x, to_reshape_back_y )


def multiply_on_first_axis(area,num):
    return area*num

v_multiply_on_first_axis= jax.vmap(multiply_on_first_axis)


def reshape_sv_labels_to_dense(tokens_a_shape, tokens_b_shape,tokens_c_shape,tokens_d_shape
                               ,tokens_a,tokens_b,tokens_c,tokens_d
                               ,label_preds
                               ,shape_reshape_cfgs
                               ):
    """ 
    given 
        list of the a b c d tokens shapes (those after reshaping to flatten but before concatenating all tokens)
        sv label prediction
        and shape reshape cgs 
    we will reconstruct the dense labels for visualization
    """
    #first we will multiply filtered masks by associated predictions

    tokens_a=v_multiply_on_first_axis(tokens_a,label_preds[0:tokens_a_shape[0]])
    tokens_b=v_multiply_on_first_axis(tokens_b,label_preds[tokens_a_shape[0]:(tokens_a_shape[0]+tokens_b_shape[0])])
    tokens_c=v_multiply_on_first_axis(tokens_c,label_preds[(tokens_a_shape[0]+tokens_b_shape[0]):(tokens_a_shape[0]+tokens_b_shape[0]+tokens_c_shape[0])])
    tokens_d=v_multiply_on_first_axis(tokens_d,label_preds[(tokens_a_shape[0]+tokens_b_shape[0]+tokens_c_shape[0]):label_preds.shape[0]])
    tokens_a=get_orgig_token_shape(tokens_a,shape_reshape_cfgs[0])
    tokens_b=get_orgig_token_shape(tokens_b,shape_reshape_cfgs[1])
    tokens_c=get_orgig_token_shape(tokens_c,shape_reshape_cfgs[2])
    tokens_d=get_orgig_token_shape(tokens_d,shape_reshape_cfgs[3])


    # recreated=list(map(lambda index,token :get_orgig_token_shape(token,shape_reshape_cfgs[index]),list(enumerate( [tokens_a,tokens_b,tokens_c,tokens_d] )) ))
    recreated_dense_label=jnp.sum(jnp.stack([tokens_a,tokens_b,tokens_c,tokens_d],axis=0),axis=0)
    return recreated_dense_label

class Simple_graph_net(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict
    edge_pairs: jnp.ndarray
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
        
        self.senders=self.edge_pairs[:,0]
        self.receivers=self.edge_pairs[:,1]
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
        return losss,tokens,label_svs






