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
import ipympl
import imageio.v3 as iio
import skimage.color
import skimage.filters
import skimage.measure
import os
from ..shape_reshape_functions import *
from functools import partial
import math
from control_points_utils import *
from geometric_sv_idea.get_areas_from_control_points.points_to_areas import *


def move_in_axis(point,weights,axis,half_r ):
    """ 
    point can move up or down axis no more than half_r from current position 
    weights indicate how strongly it shoul go down (element 0) and up the axis  
    """
    # return point.at[axis].set(point[axis]-weights[0]*half_r + weights[1]*half_r)
    return point.at[axis].set(point[axis]-weights[0]*half_r + weights[1]*half_r)

v_move_in_axis= jax.vmap(move_in_axis,in_axes=(0,0,None,None))
v_v_move_in_axis= jax.vmap(v_move_in_axis,in_axes=(0,0,None,None))


def add_single_additional_point(curried,added_points_index):
    """ 
    adding single additional point - function can be used in jax scan
    """
    main_triangle_dat,modified_control_points_coords,edge_weights,edge_weights_offset,adding_points_offset=curried
    new_edge_weights_offset=edge_weights_offset+3
    edge_weights_inner=edge_weights[edge_weights_offset:new_edge_weights_offset]
    vertex_a=modified_control_points_coords[main_triangle_dat[0,-4],:]# in first new point it would be yellow
    vertex_b=modified_control_points_coords[main_triangle_dat[0,-2],:]
    vertex_c=modified_control_points_coords[main_triangle_dat[1,-2],:]
    new_point=get_point_inside_triange(vertex_a,vertex_b,vertex_c,edge_weights_inner)
    #we modify the existing triangle data to take into account new points 
    #important we insert the same number into data about two triangles that are sharing the same edge
    main_triangle_dat= jnp.insert(main_triangle_dat, 1, adding_points_offset+added_points_index, axis=1)
    curried=main_triangle_dat,modified_control_points_coords,edge_weights,new_edge_weights_offset

    return curried,new_point

def add_new_points_per_main_triangle(main_triangle_num,triangles_data,modified_control_points_coords,edge_weights,cfg):
    """ 
    after! we had applied learned weights to the primary control points we can add more by 
    adding between gird points b and c (yellow and green)  so in primary triangles data we 
    are inserting new point between entry 0 and 1 (second) of both subtriangles of main triangle
    (here example of main traingle is BC or DL with sub triangles B,C and D,L)
    number of new points that we will add is controled in cfg num_additional_points field
    points will be created in random spot of a primary triangle for the first added point
    for the next ones the base of a triangles set on the sv centers will not chenge but
    the newly created point will replace point 0 (yellow) so each time we create new point we will
    add it in smaller triangle
    adding_points_offset - will be used for modifying main_triangle_dat so we will have later correct correspondence of 
        indicies main_triangle_dat and indicies in control_points_coords
    """
    main_triangle_dat = triangles_data[main_triangle_num,:,:]
    adding_points_offset=main_triangle_num*cfg.num_additional_points
    curried = main_triangle_dat,modified_control_points_coords,edge_weights,jnp.zeros(1),adding_points_offset
    curried,new_points=jax.lax.scan(add_single_additional_point,curried,jnp.arange(8,cfg.num_additional_points))
    main_triangle_dat,modified_control_points_coords,edge_weights,_,adding_points_offset=curried
    return new_points,main_triangle_dat

v_add_new_points_per_main_triangle= jax.vmap(add_new_points_per_main_triangle,in_axes=(0,None,None,None,None), out_axes=(0,None))

def add_new_points(triangles_data,modified_control_points_coords,edge_weights,cfg):
    """ 
    after! we had applied learned weights to the primary control points we can add more by 
    adding between gird points b and c (yellow and green)  so in primary triangles data we 
    are inserting new point between entry 0 and 1 (second) of both subtriangles of main triangle
    (here example of main traingle is BC or DL with sub triangles B,C and D,L)
    """
    new_points,main_triangle_dats=v_add_new_points_per_main_triangle(jnp.arange(4),triangles_data,modified_control_points_coords,edge_weights,cfg)
    #we integrated modified triangle data from all primary triangles
    triangles_data=jnp.stack(main_triangle_dats,axis=0)
    new_points= einops.rearrange(new_points,'a b p-> (a b) p')
    modified_control_points_coords= jnp.concatenate([modified_control_points_coords,new_points],axis=0)
    return modified_control_points_coords,triangles_data

def get_points_from_weights(grid_c_point,weights,triangles_data,half_r,cfg):
    """  
    get points around single grid_c_point
    """
    weights= nn.sigmoid(weights)

    g=grid_c_point
    sv_c_1= [g[0]-half_r,g[1]-half_r]
    sv_c_3= [g[0]+half_r,g[1]-half_r]
    sv_c_7= [g[0]-half_r,g[1]+half_r]
    sv_c_5= [g[0]+half_r,g[1]+half_r]

    modified_control_points_coords=jnp.array([ 
         get_point_inside_square(sv_c_1,sv_c_3,sv_c_5,sv_c_7,weights[0:4])#0
         ,sv_c_1#1
        ,get_point_on_a_line_b(sv_c_1,sv_c_3,weights[4])#2
        ,sv_c_3#3
        ,get_point_on_a_line_b(sv_c_3,sv_c_5,weights[5])#4
        ,sv_c_5#5
        ,get_point_on_a_line_b(sv_c_5,sv_c_7,weights[6])#6
        ,sv_c_7#7
        ,get_point_on_a_line_b(sv_c_5,sv_c_7,weights[7])#8
     ])
    
    modified_control_points_coords,triangles_data=add_new_points(triangles_data,modified_control_points_coords,weights[8:],cfg)
    return modified_control_points_coords,triangles_data

v_get_points_from_weights= jax.vmap(get_points_from_weights,in_axes=(0,0,None,None,None),out_axes=(0,None))
v_v_get_points_from_weights= jax.vmap(v_get_points_from_weights,in_axes=(0,0,None,None,None),out_axes=(0,None))


def get_points_from_weights_all(grid_c_points,weights,triangles_data,cfg):
    """ 
    grid_c_points - yellow in /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg 
    are the centers in the computations - we get as an argument non modified grid_c_points - hence on the basis of their location
    we can easily calculate positions o the surrounding sv centers and related control points 
    weights are associated with each grid c point
    """
    modified_control_points_coords,triangles_data=v_v_get_points_from_weights(grid_c_points,weights,triangles_data,cfg.r//2,cfg)
    return modified_control_points_coords,triangles_data



