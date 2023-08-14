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
from points_to_areas import *

""" 
instead of analyzing triangles and calculating all values inside we can calculate he intensity values on the lines
from sv center to its control points (creating secondary points) and between those points it will lead to not 
complete sampling of the sv area that may be more optimazation friendly

so we want to get sth like a spider web so
we need a line from a sv center to a control point - we can use triangle data for that

1) on the basis of triangle data we need to extablish the order of points so points next each other in
    data should be neighbours in the reality - also last should be neighbour with first

2) we iterate over a data and its rolled copy - so we will have in a single row info about location of 2 neighbours

3) we use infor about location of neighbours and the location of the sv center that owns this triangle
    for futher sampling

4) we get n_radius points on the line between sv center and each controll point 
    then we get n_cross points between last point of n_radius points on neighbouring radius
    and n_cross-1 points on prevoious point ...

5) we sample the value of all points  and save them in a dimension used for given sv

6) we reorder channels with points in it so we will have data of the same svs on the same channel

7) we collect all the data about single sv and calculate the variance to return it as a loss
"""

n_radius=3
n_cross=2
square_point = jnp.array([[0,0],[20,0],[20,20],[0,20]])
center = jnp.array([[10,10]])
filled_square = jnp.array(np.random.random((20,20)))

def get_radius_points(ver_a,ver_b,sv_center,weight):
    return jnp.stack([get_point_on_a_line_b(ver_a,sv_center,weight)
                      ,get_point_on_a_line_b(ver_b,sv_center,weight)
                      ])

v_get_radius_points=jax.vmap(get_radius_points,in_axes=(None,None,None,0))

def sample_in_triangle(ver_a,ver_b,sv_center, image ,n_radius,n_cross):
    pseudo_weights= jnp.arange(n_radius)
    pseudo_weights=pseudo_weights/pseudo_weights[-1]
    pseudo_weights=pseudo_weights.at[-1].set(pseudo_weights[-1]-0.1)#subtracting in order to avoid sampling the border
    radial=v_get_radius_points(ver_a,ver_b,sv_center,pseudo_weights)
    # we need to get copies of points as many as many cross points we want and a weight leading to points as distant from each other as possible


def sample_area(contr_points, sv_centers,n_radius,n_cross):
    contr_points_a=contr_points
    contr_points_b=jnp.roll(contr_points,1,0)

    print(f"ccc contr_points_b {contr_points_b}")

sample_area(square_point, center,n_radius,n_cross)