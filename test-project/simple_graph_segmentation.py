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


f = h5py.File('/workspaces/jax_cpu_experiments_b/hdf5_loc/example_mask.hdf5', 'r+')
label=f["label"][:,:]
curr_image=f["image"][:,:]
masks=f["masks"][:,:,:]
curr_image= einops.rearrange(curr_image,'w h->1 w h 1')
masks= einops.rearrange(masks,'w h c->1 w h c')


cfg=get_cfg()
initial_masks= jnp.stack([
    get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
    get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
    get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
    get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
        ],axis=0)
initial_masks=jnp.sum(initial_masks,axis=0)   

print(f"curr_image {curr_image.shape} masks {masks.shape}  initial_masks {initial_masks.shape} ")



def work_on_single_area(curr_id,mask_curr,image):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    masked_image= jnp.multiply(image,filtered_mask)
    dummy_token= jnp.zeros(5)+jnp.mean(masked_image)
    return dummy_token

v_work_on_single_area=jax.vmap(work_on_single_area)
v_v_work_on_single_area=jax.vmap(v_work_on_single_area)


def iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    curr_image_in=divide_sv_grid(curr_image,shape_reshape_cfg)
    tokens= v_v_work_on_single_area(curr_ids,mask_curr,curr_image_in)


    return tokens



initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')

shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)
curr_image_out_meaned= np.zeros_like(curr_image)

for i in range(4):        
    tokens = iter_over_masks(shape_reshape_cfgs,i,masks,curr_image,shape_reshape_cfgs_old,initial_masks)
    

print(f"curr_image_out_meaned {curr_image_out_meaned.shape}")    



""" 
we have the masks and image now on the basis of those we will do the projection of each mask
    this will represent the token for each node
we need also to get the edges for simplicity we will just the nearest left - right, top - bottom neighbours
    
"""