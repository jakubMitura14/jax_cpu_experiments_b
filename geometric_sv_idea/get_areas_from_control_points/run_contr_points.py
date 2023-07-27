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

r=8
num_additional_points=1
half_r=r/2
diam_x=64 +r#256+r
diam_y=64 +r#256+r
gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r
grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]

# cfg={'r': r,'num_additional_points':num_additional_points}

cfg = config_dict.ConfigDict()
cfg.r=r
cfg.num_additional_points=num_additional_points
cfg = ml_collections.config_dict.FrozenConfigDict(cfg)

sv_diameter=r
pmapped_batch_size=1

weights=np.random.random((pmapped_batch_size,grid_c_points.shape[0],grid_c_points.shape[1],8+num_additional_points*3))
# weights=(np.random.random((grid_a_points.shape[0],grid_a_points.shape[1],8)))/2




############# end points display








grid_c_points=einops.repeat(grid_c_points, 'x y c->b x y c', b=pmapped_batch_size) 

debug_index=0
# for debug_index in range(13824):

num_additional_points=3
primary_control_points_offset=9
triangles_data=get_modified_triangles_data(num_additional_points,primary_control_points_offset)

modified_control_points_coords=batched_get_points_from_weights_all(grid_c_points,weights,r,num_additional_points,triangles_data)
ress=analyze_all_control_points(modified_control_points_coords,triangles_data
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


# print(f"debug_index {debug_index} count {sum_count}  minn {jnp.min(ress.flatten())} maxx {jnp.max(ress.flatten())} ")
# # plt.show()
# plt.clf()

# diam_x=32 #256+r
# diam_y=32 #256+r
# img_size=(1,diam_x,diam_y)
# orig_grid_shape=grid_a_points.shape

# sh_re_consts=get_simple_sh_resh_consts(img_size,r)
# disp= list(map(lambda i: reshape_to_svs(ress,sh_re_consts[i],i) ,range(4)))
# disp= jnp.concatenate(disp,axis=1)

# for i in range(disp.shape[1]):
#     sns.heatmap(disp[0,i,:,:])
#     path= f"/workspaces/jax_cpu_experiments_b/explore/debuggin_reshuffle_channels/subm{i}.png"
#     plt.savefig(path)
#     plt.clf()
# #     # plt.show()
    





""" 
sizee= (r*2)


for mask 0 
beg_pad_x = r+int(r//2)+1
beg_pad_y = r+int(r//2)+1

for mask 1 
beg_pad_x = int(r//2)+1
beg_pad_y = r+int(r//2)+1

for mask 2
beg_pad_x = int(r//2)+1
beg_pad_y = int(r//2)+1

for mask 3 
beg_pad_x = r+int(r//2)+1
beg_pad_y = int(r//2)+1

"""



