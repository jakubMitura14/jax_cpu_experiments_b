#based on https://github.com/yuanqqq/SIN
from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import jax
from ml_collections import config_dict
from functools import partial

from flax.linen import partitioning as nn_partitioning
import pandas as pd
# remat = nn_partitioning.remat
import ml_collections

def get_diameter_no_pad(r):
    """
    so every time we have n elements we can get n+ more elements
    so analyzing on single axis
    start from 1 ->1+1+1 =3 good
    start from 3 ->3+3+1=7 good 
    start from 7 ->7+7+1=15 good 
    """
    curr = 1
    for i in range(0,r):
        curr=curr*2+1
    return curr

def get_diameter(r):
    return get_diameter_no_pad(r)+1





def work_on_single_area_for_init(curr_id,shape_reshape_cfg,mask_curr):
    
    filtered_mask=mask_curr[:,:,curr_id]
    dima_x=(shape_reshape_cfg.diameter_x)//2
    dima_y=(shape_reshape_cfg.diameter_y)//2

    dima_x_half=dima_x//2
    dima_y_half=dima_y//2


    filtered_mask=filtered_mask.at[dima_x-dima_x_half:dima_x+dima_x_half
                                    ,dima_y-dima_y_half:dima_y+dima_y_half].set(1)
    filtered_mask=einops.rearrange(filtered_mask,'w h-> w h 1')
    return filtered_mask

v_work_on_single_area_for_init=jax.vmap(work_on_single_area_for_init,in_axes=(None,None,0))
v_v_work_on_single_area_for_init=jax.vmap(v_work_on_single_area_for_init,in_axes=(None,None,0))

def iter_over_masks_for_init(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old):
    shape_reshape_cfg=shape_reshape_cfgs[i]
    shape_reshape_cfg_old=shape_reshape_cfgs_old[i]
    # curr_ids=initial_masks[:,shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2,: ]
    # curr_ids=einops.rearrange(curr_ids,'b x y p ->b (x y) p')
    mask_curr=divide_sv_grid(masks,shape_reshape_cfg)
    # shapee_edge_diff=curr_image.shape
    masked_image= v_v_work_on_single_area_for_init(i,shape_reshape_cfg,mask_curr)

    to_reshape_back_x=np.floor_divide(shape_reshape_cfg.axis_len_x,shape_reshape_cfg.diameter_x)
    to_reshape_back_y=np.floor_divide(shape_reshape_cfg.axis_len_y,shape_reshape_cfg.diameter_y) 

    to_reshape_back_x_old=np.floor_divide(shape_reshape_cfg_old.axis_len_x,shape_reshape_cfg_old.diameter_x)
    to_reshape_back_y_old=np.floor_divide(shape_reshape_cfg_old.axis_len_y,shape_reshape_cfg_old.diameter_y) 

    masked_image=recreate_orig_shape(masked_image,shape_reshape_cfg,to_reshape_back_x,to_reshape_back_y )

    return masked_image

def get_init_masks(cfg):  
    r_x_total=cfg.r_x_total
    r_y_total=cfg.r_y_total
    masks= jnp.zeros((1,cfg.img_size[1],cfg.img_size[2],cfg.num_dim))
    shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=r_x_total,r_y=r_y_total)
    res= list(map(lambda i :iter_over_masks_for_init(shape_reshape_cfgs,i,masks,shape_reshape_cfgs_old)[0,:,:,0], range(4)))
    return jnp.stack(res,axis=-1)


def get_initial_supervoxel_masks(orig_grid_shape,shift_x,shift_y,mask_num):
    """
    on the basis of the present shifts we will initialize the masks
    ids of the supervoxels here are implicit based on which mask and what location we are talking about
    """
    initt=np.zeros(orig_grid_shape)
    initt[shift_x::2,shift_y::2,mask_num]=1
    return initt




@partial(jax.jit, static_argnames=['diameter_x','diameter_y','p_x','p_y'])
def set_non_overlapping_regions(diameter_x:int
                                ,diameter_y:int
                                ,shift_x:int
                                ,shift_y:int
                                ,p_x:int
                                ,p_y:int
                                ):
    """
    sets non overlapping regions of each mask to 1
    """
    # shift_x= jnp.remainder(index,2)
    # shift_y=index//2
    # p_x=jnp.maximum(((diameter_x-1)//2)-1)#-shape_reshape_cfg.shift_x
    # p_y=jnp.maximum(((diameter_y-1)//2)-1)#-shape_reshape_cfg.shift_y
    s_x=shift_x
    s_y=shift_y
    # return jnp.zeros((diameter_x,diameter_y)).at[p_x+s_x:-(p_x-s_x),p_y+s_y:-(p_y-s_y)].set(1)
    beg_x=p_x+s_x
    end_x=(p_x-s_x)
    beg_y=p_y+s_y
    end_y= (p_y-s_y)
    oness = jnp.ones((diameter_x-p_x*2,diameter_y-p_y*2))
    return jax.lax.dynamic_update_slice(jnp.zeros((diameter_x,diameter_y )), oness, (beg_x,beg_y))
    # return jnp.pad(oness,((beg_x,end_x),(beg_y,end_y)))
    # return jnp.pad(oness,(jnp.stack([beg_x,end_x]),jnp.stack([beg_y,end_y])))

def for_pad_divide_grid(current_grid_shape:Tuple[int],axis:int,shift:int,orig_grid_shape:Tuple[int],diameter:int):
    """
    helper function for divide_sv_grid in order to calculate padding
    additionally give the the right infor for cut
    """
    #calculating the length of the axis after all of the cuts and paddings
    #for example if we have no shift we need to add r at the begining of the axis
    # r_to_pad=(get_diameter_no_pad(r)-1)//2
    r_to_pad=(diameter+1)//2
    # r_to_pad=(get_diameter_no_pad(r))//2
    # print(f"get_diameter_no_pad(r) {get_diameter_no_pad(r)}")

    for_pad_beg=r_to_pad*(1-shift)
    #wheather we want to remove sth from end or not depend wheater we have odd or even amountof supervoxel ids in this axis
    is_even=int((orig_grid_shape[axis]%2==0))
    is_odd=1-is_even
    to_remove_from_end= (shift*is_odd)*r_to_pad + ((1-shift)*is_even)*r_to_pad
    axis_len_prim=for_pad_beg+current_grid_shape[axis]-to_remove_from_end
    #how much padding we need to make it divisible by diameter
    for_pad_rem= np.remainder(axis_len_prim,diameter)
    to_pad_end=diameter-np.remainder(axis_len_prim,diameter)

    if(for_pad_rem==0):
        to_pad_end=0

    # f1 = lambda: 0
    # f2 = lambda: to_pad_end
    # to_pad_end=jax.lax.cond(for_pad_rem==0,f1,f2)
 
    axis_len=axis_len_prim+to_pad_end    
    return for_pad_beg,to_remove_from_end,axis_len_prim,axis_len,to_pad_end     

def get_supervoxel_ids(shape_reshape_cfg):
    """
    In order to be able to vmap through the supervoxels we need to have a way 
    to tell what id should be present in the area we have and that was given by main part of 
    divide_sv_grid function the supervoxel ids are based on the orig_grid_shape  generally 
    we have the supervoxel every r but here as we jump every 2r we need every second id
    """
    # shape_reshape_cfg=array_toshape_reshape_constants(shape_reshape_cfg_arr)
    res_grid=jnp.mgrid[1:shape_reshape_cfg.orig_grid_shape[0]+1, 1:shape_reshape_cfg.orig_grid_shape[1]+1]
    res_grid=einops.rearrange(res_grid,'p x y-> x y p')
    res_grid= res_grid[shape_reshape_cfg.shift_x: shape_reshape_cfg.orig_grid_shape[0]:2,
                    shape_reshape_cfg.shift_y: shape_reshape_cfg.orig_grid_shape[1]:2, ]
    
    return einops.rearrange(res_grid,'x y p -> (x y) p')                 

def divide_sv_grid(res_grid: jnp.ndarray,shape_reshape_cfg):
    """
    as the supervoxel will overlap we need to have a way to divide the array with supervoxel ids
    into the set of non overlapping areas - we want thos area to be maximum possible area where we could find
    any voxels associated with this supervoxels- the "radius" of this cube hence can be calculated based on the amount of dilatations made
    becouse of this overlapping we need to be able to have at least 8 diffrent divisions
    we can work them out on the basis of the fact where we start at each axis at 0 or r - and do it for
    all axis permutations 2**3 =8
    we need also to take care about padding after removing r from each axis the grid need to be divisible by 2*r+1
    as the first row and column do not grow back by construction if there is no shift we always need to add r padding rest of pad to the end
    in case no shift is present all padding should go at the end
    """
    print(f"initial res_grid {res_grid.shape}")

    cutted=res_grid[:,0: shape_reshape_cfg.curr_image_shape[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.curr_image_shape[1]- shape_reshape_cfg.to_remove_from_end_y,:]
    print(f"pre_pad cutted {cutted.shape}")
    cutted= jnp.pad(cutted,((0,0),
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ,(0,0)))
    print(f"post_pad cutted {cutted.shape}")

    print(f"sssshape_reshape_cfg.to_pad_beg_x {shape_reshape_cfg.to_pad_beg_x}")

    cutted=einops.rearrange( cutted,'bb (a x) (b y) cc->bb (a b) x y cc', x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    return cutted


def divide_sv_grid_no_batch(res_grid: jnp.ndarray,shape_reshape_cfg):

    cutted=res_grid[0: shape_reshape_cfg.curr_image_shape[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.curr_image_shape[1]- shape_reshape_cfg.to_remove_from_end_y,:]
    cutted= jnp.pad(cutted,(
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,'(a x) (b y) cc->(a b) x y cc', x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    return cutted



def divide_sv_grid_p_mapped(res_grid: jnp.ndarray,shape_reshape_cfg):

    cutted=res_grid[:,:,0: shape_reshape_cfg.curr_image_shape[0]- shape_reshape_cfg.to_remove_from_end_x
                    ,0: shape_reshape_cfg.curr_image_shape[1]- shape_reshape_cfg.to_remove_from_end_y,:]
    cutted= jnp.pad(cutted,((0,0),(0,0),
                        (shape_reshape_cfg.to_pad_beg_x,shape_reshape_cfg.to_pad_end_x)
                        ,(shape_reshape_cfg.to_pad_beg_y,shape_reshape_cfg.to_pad_end_y )
                        ,(0,0)))
    cutted=einops.rearrange( cutted,'pp bb (a x) (b y) cc->pp bb (a b) x y cc', x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    return cutted

def recreate_orig_shape(texture_information: jnp.ndarray,shape_reshape_cfg
                        ,to_reshape_back_x:int
                        ,to_reshape_back_y:int):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    # undo axis reshuffling
    texture_information= einops.rearrange(texture_information,'bb (a b) x y cc->bb (a x) (b y) cc'
        ,a=to_reshape_back_x
        ,b=to_reshape_back_y
        ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange(texture_information,'bb (a b) x y->bb (a x) (b y)'
    #     ,a=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter_x
    #     ,b=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter_y
    #     ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange( texture_information,'a x y->(a x y)')
    #undo padding
    texture_information= texture_information[:,
            shape_reshape_cfg.to_pad_beg_x: shape_reshape_cfg.axis_len_x- shape_reshape_cfg.to_pad_end_x
            ,shape_reshape_cfg.to_pad_beg_y:shape_reshape_cfg.axis_len_y- shape_reshape_cfg.to_pad_end_y,:]
   
    #undo cutting
    texture_information= jnp.pad(texture_information,((0,0),
                        (0,shape_reshape_cfg.to_remove_from_end_x)
                        ,(0,shape_reshape_cfg.to_remove_from_end_y )
                        ,(0,0)))
    return texture_information

def recreate_orig_shape_simple(texture_information: jnp.ndarray,shape_reshape_cfg):
    """
    as in divide_sv_grid we are changing the shape for supervoxel based texture infrence
    we need then to recreate undo padding axis reshuffling ... to get back the original image shape
    """
    # undo axis reshuffling
    # texture_information= einops.rearrange(texture_information,'bb (a b) x y cc->bb (a x) (b y) cc'
    #     ,a=to_reshape_back_x
    #     ,b=to_reshape_back_y
    #     ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    texture_information= einops.rearrange(texture_information,'bb (a b) x y c->bb (a x) (b y) c'
        ,a=shape_reshape_cfg.axis_len_x//shape_reshape_cfg.diameter_x
        ,b=shape_reshape_cfg.axis_len_y//shape_reshape_cfg.diameter_y
        ,x=shape_reshape_cfg.diameter_x,y=shape_reshape_cfg.diameter_y)
    # texture_information= einops.rearrange( texture_information,'a x y->(a x y)')
    #undo padding
    texture_information= texture_information[:,
            shape_reshape_cfg.to_pad_beg_x: shape_reshape_cfg.axis_len_x- shape_reshape_cfg.to_pad_end_x
            ,shape_reshape_cfg.to_pad_beg_y:shape_reshape_cfg.axis_len_y- shape_reshape_cfg.to_pad_end_y,:]
   
    #undo cutting
    texture_information= jnp.pad(texture_information,((0,0),
                        (0,shape_reshape_cfg.to_remove_from_end_x)
                        ,(0,shape_reshape_cfg.to_remove_from_end_y )
                        ,(0,0)))
    return texture_information


def get_shape_reshape_constants(shift_x:bool,shift_y:bool, r_x:int, r_y:int,img_size ,orig_grid_shape):
    """
    provides set of the constants required for reshaping into non overlapping areas
    what will be used to analyze supervoxels separately 
    results will be saved in a frozen configuration dict
    """
    diameter_x=r_x
    diameter_y=r_y
    curr_image_shape= (img_size[1],img_size[2])
    # shift_x=int(shift_x)
    # shift_y=int(shift_y)
    to_pad_beg_x,to_remove_from_end_x,axis_len_prim_x,axis_len_x,to_pad_end_x  =for_pad_divide_grid(curr_image_shape,0,shift_x,orig_grid_shape,diameter_x)
    to_pad_beg_y,to_remove_from_end_y,axis_len_prim_y,axis_len_y,to_pad_end_y   =for_pad_divide_grid(curr_image_shape,1,shift_y,orig_grid_shape,diameter_y)

    res_cfg = config_dict.ConfigDict()
    res_cfg.to_pad_beg_x=to_pad_beg_x
    res_cfg.to_remove_from_end_x=to_remove_from_end_x
    res_cfg.axis_len_prim_x=axis_len_prim_x
    res_cfg.axis_len_x=axis_len_x
    res_cfg.to_pad_beg_y=to_pad_beg_y
    res_cfg.to_remove_from_end_y=to_remove_from_end_y
    res_cfg.axis_len_prim_y=axis_len_prim_y
    res_cfg.axis_len_y=axis_len_y
    res_cfg.to_pad_end_x=to_pad_end_x
    res_cfg.to_pad_end_y=to_pad_end_y
    res_cfg.shift_x=shift_x
    res_cfg.shift_y=shift_y
    res_cfg.orig_grid_shape=orig_grid_shape
    res_cfg.diameter_x=diameter_x
    res_cfg.diameter_y=diameter_y
    res_cfg.img_size=img_size
    res_cfg.curr_image_shape=curr_image_shape
    res_cfg = ml_collections.config_dict.FrozenConfigDict(res_cfg)

    return res_cfg



def get_all_shape_reshape_constants(r_x:int,r_y:int,img_size ,orig_grid_shape ):
    """
    gives the shape reshape config objects for all shifts configurtations
    cfgs in order are for shift_x,shift_y
    0) 0 0
    1) 1 0
    2) 0 1
    3) 1 1
    """
    return[get_shape_reshape_constants(shift_x=0,shift_y=0, r_x=r_x, r_y=r_y,img_size=img_size ,orig_grid_shape=orig_grid_shape )
           ,get_shape_reshape_constants(shift_x=1,shift_y=0, r_x=r_x, r_y=r_y,img_size=img_size ,orig_grid_shape=orig_grid_shape )
           ,get_shape_reshape_constants(shift_x=0,shift_y=1, r_x=r_x, r_y=r_y,img_size=img_size ,orig_grid_shape=orig_grid_shape )
           ,get_shape_reshape_constants(shift_x=1,shift_y=1, r_x=r_x, r_y=r_y ,img_size=img_size ,orig_grid_shape=orig_grid_shape)]
    # arr=[shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=0,shift_y=0, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=1,shift_y=0, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=0,shift_y=1, r_x=r_x, r_y=r_y ))
    #        ,shape_reshape_constants_to_array(get_shape_reshape_constants(cfg,shift_x=1,shift_y=1, r_x=r_x, r_y=r_y ))]
    # return jnp.stack(arr)


def disp_to_pandas(probs,shappe ):
    probs_to_disp= einops.rearrange(probs,'w h c-> (w h) c')
    probs_to_disp=np.round(probs_to_disp,1)
    # probs_to_disp=list(map(lambda twoo: f"{twoo[0]} {twoo[1]}",list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp)
    probs_to_disp=list(map(lambda twoo: " ".join(list(map(str,twoo))),list(probs_to_disp)))
    probs_to_disp=np.array(probs_to_disp).reshape(shappe)
    return pd.DataFrame(probs_to_disp)

def disp_to_pandas_curr_shape(probs ):
    return disp_to_pandas(probs,(probs.shape[0],probs.shape[1]) )

