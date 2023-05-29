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
import jraph

f = h5py.File('/workspaces/jax_cpu_experiments_b/hdf5_loc/example_mask.hdf5', 'r+')
label=f["label"][:,:]
curr_image=f["image"][:,:]
masks=f["masks"][:,:,:]
curr_image= einops.rearrange(curr_image,'w h->1 w h 1')
masks= einops.rearrange(masks,'w h c->1 w h c')



def work_on_single_area(curr_id,mask_curr,image):
    filtered_mask=filter_mask_of_intrest(mask_curr,curr_id)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    masked_image= jnp.multiply(image,filtered_mask)
    # dummy_token= jnp.zeros(5)+jnp.mean(masked_image)
    return masked_image

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
    tokens=einops.rearrange(tokens,'b f x y c->(b f) x y c')
    return tokens



class Simple_graph_net(nn.Module):
    cfg: ml_collections.config_dict.config_dict.ConfigDict

    def setup(self):
        initial_masks= jnp.stack([
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
            get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
                ],axis=0)
        initial_masks=jnp.sum(initial_masks,axis=0)   
        initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
        self.shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
        self.shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)
        edge_pairs=get_sorce_targets(cfg.orig_grid_shape)
        self.senders=edge_pairs[:,0]
        self.receivers=edge_pairs[:,1]
        self.n_node = initial_masks.shape[1]*initial_masks.shape[2]
        self.n_edge = self.senders.shape[0]

    
    @nn.compact
    def __call__(self, curr_image: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
        tokens_a = iter_over_masks(self.shape_reshape_cfgs,0,masks,curr_image,self.shape_reshape_cfgs_old,self.initial_masks)
        tokens_b = iter_over_masks(self.shape_reshape_cfgs,1,masks,curr_image,self.shape_reshape_cfgs_old,self.initial_masks)
        tokens_c = iter_over_masks(self.shape_reshape_cfgs,2,masks,curr_image,self.shape_reshape_cfgs_old,self.initial_masks)
        tokens_d = iter_over_masks(self.shape_reshape_cfgs,3,masks,curr_image,self.shape_reshape_cfgs_old,self.initial_masks)
        tokens= jnp.concatenate([tokens_a,tokens_b,tokens_c,tokens_d ])
        print(f"ttttttttttt tokens {tokens.shape}")
        tokens=Conv_trio(self.cfg,40)(tokens)
        tokens=Conv_trio(self.cfg,40)(tokens)
        tokens= einops.rearrange(tokens,'f x y c-> f (x y c)')
        tokens= nn.Dense(features=6)(tokens)
        # Optionally you can add `global` information, such as a graph label.
        global_context = jnp.array([[1]]) # Same feature dims as nodes and edges.
        graph = jraph.GraphsTuple(
            nodes=tokens,
            edges=self.edges,
            senders=self.senders,
            receivers=self.receivers,
            n_node=self.n_node,
            n_edge=self.n_edge,
            globals=global_context
      )





cfg=get_cfg()





print(tokens_a.shape)
print(tokens.shape)

""" 
we have the masks and image now on the basis of those we will do the projection of each mask
    this will represent the token for each node
we need also to get the edges for simplicity we will just the nearest left - right, top - bottom neighbours
    
"""