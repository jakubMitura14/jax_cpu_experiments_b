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
from shape_reshape_functions import *
from functools import partial
import math
from set_points_loc import *
from points_to_areas import *
from integrate_triangles import *
import integrate_triangles
from jax.config import config

config.update("jax_debug_nans", True)

r=16
num_additional_points=3
primary_control_points_offset=9
half_r=r/2
diam_x=128 +r#256+r
diam_y=128 +r#256+r
gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r
grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]

# cfg={'r': r,'num_additional_points':num_additional_points}

cfg = config_dict.ConfigDict()
cfg.r=r
cfg.num_additional_points=num_additional_points
cfg.primary_control_points_offset=primary_control_points_offset
cfg = ml_collections.config_dict.FrozenConfigDict(cfg)

sv_diameter=r
pmapped_batch_size=1

weights=np.random.random((pmapped_batch_size,grid_c_points.shape[0],grid_c_points.shape[1],6+num_additional_points*3))/2
# weights=np.ones_like(weights)/2#TODO remove
# weights=(np.random.random((grid_a_points.shape[0],grid_a_points.shape[1],8)))/2




grid_c_points=einops.repeat(grid_c_points, 'x y c->b x y c', b=pmapped_batch_size) 

debug_index=0
# for debug_index in range(13824):


triangles_data=get_triangles_data()
triangles_data_modif=integrate_triangles.get_modified_triangles_data(num_additional_points,primary_control_points_offset)
# triangles_data_modif=einops.rearrange(triangles_data,'a b p-> (a b) p')
# # print(triangles_data)
modified_control_points_coords=batched_get_points_from_weights_all(grid_c_points,weights,r,num_additional_points,triangles_data)

def disp_grid(modified_control_points_coords):
    modified_control_points_coords=modified_control_points_coords[0,:,:,:,:]
    modified_control_points_coords=einops.rearrange(modified_control_points_coords,'x y t p-> t x y p')
    shh=modified_control_points_coords.shape
    # colors=np.array(['aqua','beige','black','blue','brown','coral','fuschsia','gold','green','indigo','magneta','purple'])
    colors=np.array(['yellow'#0
                     ,'black'#1
                     ,'green'#2
                     ,'black'#3
                     ,'green'#4
                     ,'brown'#5 black
                     ,'green'#6
                     ,'black'#7
                     ,'green'#8


                     ,'red'#9
                     ,'blue'#10
                     
                     ,'red'#11
                     ,'gold'#12
                     
                     ,'red'#13
                     ,'grey'#14
                     
                     ,'red'#15
                     ,'purple'#16

                     ])
    colors=einops.repeat(colors,'a-> (a b)' ,b=shh[1]*shh[2])
    s= np.ones((shh[0],shh[1],shh[2]))*12
    s[:,1,1]=70

    x= modified_control_points_coords[:,:,:,0].flatten()
    y= modified_control_points_coords[:,:,:,1].flatten()
    plt.scatter(x,y,s=s.flatten(),color=colors,alpha=0.7)
    plt.savefig('/workspaces/jax_cpu_experiments_b/explore/points.png')


print(f"modified_control_points_coords {modified_control_points_coords.shape} grid_c_points {grid_c_points.shape}")
# disp_grid(modified_control_points_coords)

ress=analyze_all_control_points(modified_control_points_coords,triangles_data_modif
                               ,pmapped_batch_size,sv_diameter,half_r,cfg.num_additional_points)
print(ress.shape)

mask0=ress[0,:,:,0]
# sns.heatmap(mask0,ax=axs[0])
# plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask0.png')
# plt.show()
mask1=ress[0,:,:,1]
# sns.heatmap(mask1)
# plt.show()
# plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask1.png')


fig, axs = plt.subplots(nrows=2,ncols=2)

sns.heatmap(mask0,ax=axs[0,0]).legend([],[], frameon=False)
sns.heatmap(mask1,ax=axs[0,1]).legend([],[], frameon=False)
sns.heatmap(ress[0,:,:,2],ax=axs[1,0]).legend([],[], frameon=False)
sns.heatmap(ress[0,:,:,3],ax=axs[1,1]).legend([],[], frameon=False)

# file_name=f"/workspaces/jax_cpu_experiments_b/explore/debuggin_reshuffle_channels/example_mask{debug_index}.png"
file_name=f"/workspaces/jax_cpu_experiments_b/explore/all_masks.png"
plt.savefig(file_name)
plt.clf()
file_name=f"/workspaces/jax_cpu_experiments_b/explore/all_masks_sum.png"
sns.heatmap(jnp.sum(ress[0,:,:,:],axis=-1)).legend([],[], frameon=False)
plt.savefig(file_name)
plt.clf()
