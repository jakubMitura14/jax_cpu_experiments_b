import numpy as np
import matplotlib.pyplot as plt
import toolz
import optax
import jax.numpy as jnp
import jax
from flax import linen as nn
import einops
import seaborn as sns
from itertools import permutations
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import skimage.color
import skimage.filters
import skimage.measure
import os
from shape_reshape_functions import *
from functools import partial
import math
import itertools
import points_to_areas
from points_to_areas import *



def get_radius_points(ver_a,ver_b,sv_center,weight):
    return jnp.stack([get_point_on_a_line_b(ver_a,sv_center,weight)
                      ,get_point_on_a_line_b(ver_b,sv_center,weight)
                      ])

v_get_radius_points=jax.vmap(get_radius_points,in_axes=(None,None,None,0))

def get_radial_weight(num_p):
    w=np.arange(num_p+2)
    w=w/w[-1]
    return w[1:-1]


def get_indicies_for_cross(n_radius,n_cross):
    indicies= jnp.arange(n_radius)
    to_repeat=jnp.pad(jnp.arange(n_cross,0,-1),(0,n_radius-n_cross))
    indicies= jnp.repeat(indicies,to_repeat)
    weights= jnp.array(list(itertools.chain(*list(map(get_radial_weight, to_repeat)))))
    return (indicies,weights)

def get_cross_points(radial,index,weight):
    return get_point_on_a_line_b(radial[index,0,:],radial[index,1,:],weight)

v_get_cross_points=jax.vmap(get_cross_points,in_axes=(None,0,0))

def sample_in_triangle(ver_a,ver_b,sv_center, image ,n_radius,n_cross,indicies_for_cross,weights_for_cross):
    pseudo_weights= jnp.arange(n_radius+1)
    # pseudo_weights=jnp.power(pseudo_weights,1.6)
    pseudo_weights=pseudo_weights/pseudo_weights[-1]
    pseudo_weights=pseudo_weights.at[0].set(pseudo_weights[0]-0.05)#subtracting in order to avoid sampling the border
    pseudo_weights=pseudo_weights[1:]
    radial=v_get_radius_points(ver_a,ver_b,sv_center,pseudo_weights)
    # we need to get copies of points as many as many cross points we want and a weight leading to points as distant from each other as possible
    cross_points=v_get_cross_points(radial,indicies_for_cross,weights_for_cross )
    radial= einops.rearrange(radial,'a b p-> (a b) p')

    # dist=(jnp.sum(jnp.dot(ver_a,sv_center).flatten())+jnp.sum(jnp.dot(ver_b,sv_center).flatten()))/2
    # dist=(jnp.sqrt(jnp.sum(jnp.dot(ver_a,sv_center).flatten()))+jnp.sqrt(jnp.sum(jnp.dot(ver_b,sv_center).flatten())))/2
    dist=(jnp.sum(jnp.abs(ver_a-sv_center)).flatten()+jnp.sum(jnp.abs(ver_b-sv_center)).flatten())/2

    #adding also the sv center   
    sv_center=jnp.expand_dims(sv_center,axis=0)
    res= jnp.concatenate([radial ,cross_points,sv_center ],axis=0)
    return res,dist

v_sample_in_triangle=jax.vmap(sample_in_triangle,in_axes=(0,0,None,None,None,None,None,None))







def get_sv_area_type(modified_control_points_coords,pmapped_batch_size):
    """
    in order to provide th consistency of sv identification between neighbouring svs we need to 
    identify diffrent superpixel area types like in triangle geometric sv image alfa beta delta and gamma
    """
    shh=modified_control_points_coords.shape
    sv_area_type=jnp.zeros((shh[1],shh[2]),dtype=int)
    sv_area_type=sv_area_type.at[1::2,0::2].set(3)
    sv_area_type=sv_area_type.at[1::2,1::2].set(2)
    sv_area_type=sv_area_type.at[0::2,1::2].set(1)
    sv_area_type=einops.repeat(sv_area_type, 'x y->b x y', b=pmapped_batch_size)
    return sv_area_type


def sample_area(contr_points, sv_center,n_radius,n_cross,image,indicies_for_cross,weights_for_cross):
    contr_points_a=contr_points
    contr_points_b=jnp.roll(contr_points,1,0)
    # sv_center=sv_centers[0,:]#TODO change
    res,dist=v_sample_in_triangle(contr_points_a,contr_points_b,sv_center, image ,n_radius,n_cross,indicies_for_cross,weights_for_cross)
    # res=einops.rearrange(res,'a b p-> (a b) p')
    return res,dist

v_sample_area=jax.vmap(sample_area,in_axes=(0,))

def map_coords_and_norm(sampled_points_coords,image_new,distt,meann):
    sampled_points_coords=einops.rearrange(sampled_points_coords,'a b -> b a')
    sampled_points=jax.scipy.ndimage.map_coordinates(image_new,sampled_points_coords,order=1)
    sampled_points=(sampled_points-meann)*distt
    sampled_points= jnp.power(sampled_points,2)
    return sampled_points.flatten()



def sample_all_contr_points(modified_control_points_coords,pmapped_batch_size,image,half_r
                                ,n_radius,n_cross,indicies_for_cross,weights_for_cross ):
    """ 
    will get all modified_control_points_coords and sample the underyling areas 
    function analogous to analyze_all_control_points
    """
    sv_area_type=get_sv_area_type(modified_control_points_coords,pmapped_batch_size)
    #we need to pad the image so it will fit the modified_control_points_coords (those are yellow control points centered)
    image = jnp.pad(image,((0,0),(half_r,half_r),(half_r,half_r),(0,0)))
    #now we need to vmap over all areas 



