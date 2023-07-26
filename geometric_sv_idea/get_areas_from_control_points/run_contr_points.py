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


r=8
num_additional_points=1
half_r=r/2
diam_x=64 +r#256+r
diam_y=64 +r#256+r
gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r
grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]
cfg={'r': r,'num_additional_points':num_additional_points}


weights=np.random.random((grid_c_points.shape[0],grid_c_points.shape[1],8+num_additional_points*3))
# weights=(np.random.random((grid_a_points.shape[0],grid_a_points.shape[1],8)))/2

# weights= jnp.ones_like(weights)
# weights=weights.at[1,1,:].set(3.0)





####################3 end random point mutation
###########3just points display


def disp_grid(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points):

    c_a=np.ones_like(grid_a_points[:,:,1])-0.9
    c_b_x=np.ones_like(grid_b_points_x[:,:,1])+0.9
    c_b_y=np.ones_like(grid_b_points_y[:,:,1])+0.9
    c_c=np.ones_like(grid_c_points[:,:,1])+2.0


    s_a=np.ones_like(grid_a_points[:,:,1])
    s_b_x=np.ones_like(grid_b_points_x[:,:,1])
    s_b_y=np.ones_like(grid_b_points_y[:,:,1])
    s_c=np.ones_like(grid_c_points[:,:,1])
    base_x=2
    base_y=2

    # s_a[base_x,base_y]=s_a[0,0]*4
    # s_b_x[base_x,base_y]=s_b_x[0,0]*4
    # s_b_y[base_x,base_y]=s_b_y[0,0]*4
    # s_b_x[base_x+1,base_y]=s_b_x[0,0]*4
    # s_b_y[base_x,base_y+1]=s_b_y[0,0]*4

    # s_c[base_x,base_y]=s_c[0,0]*4
    # s_c[base_x,base_y+1]=s_c[0,0]*4
    # s_c[base_x+1,base_y]=s_c[0,0]*4
    # s_c[base_x+1,base_y+1]=s_c[0,0]*4


    grid_b_points_x=einops.rearrange(grid_b_points_x,'x y c-> (x y) c')
    grid_b_points_y=einops.rearrange(grid_b_points_y,'x y c-> (x y) c')
    grid_c_points=einops.rearrange(grid_c_points,'x y c-> (x y) c')
    grid_a_points=einops.rearrange(grid_a_points,'x y c-> (x y) c')

    grid_b_points= jnp.concatenate([grid_b_points_x,grid_b_points_y])
    x=jnp.concatenate([grid_a_points[:,0],grid_b_points[:,0],grid_c_points[:,0]])
    y=jnp.concatenate([grid_a_points[:,1],grid_b_points[:,1],grid_c_points[:,1]])

    c= jnp.concatenate([c_a.flatten(),c_b_x.flatten(),c_b_y.flatten(),c_c.flatten()])
    s=jnp.concatenate([s_a.flatten(),s_b_x.flatten(),s_b_y.flatten(),s_c.flatten()])*15
    plt.scatter(x,y,s=s,c=c,alpha=0.7)
    file_name=f"/workspaces/jax_cpu_experiments_b/explore/points.png"
    plt.savefig(file_name)
    plt.clf()


disp_grid(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points)








############# end points display







sv_diameter=r
pmapped_batch_size=1
grid_a_points=einops.repeat(grid_a_points, 'x y c->b x y c', b=pmapped_batch_size) 
grid_b_points_x=einops.repeat(grid_b_points_x, 'x y c->b x y c', b=pmapped_batch_size) 
grid_b_points_y=einops.repeat(grid_b_points_y, 'x y c->b x y c', b=pmapped_batch_size) 
grid_c_points=einops.repeat(grid_c_points, 'x y c->b x y c', b=pmapped_batch_size) 






def connected_components(image, sigma=1.0, t=0.0001, connectivity=2):
    # load the image
    # image = iio.imread(filename)
    # mask the image according to threshold
    binary_mask = image >0
    # perform connected component analysis
    labeled_image, count = skimage.measure.label(binary_mask,
                                                 connectivity=connectivity, return_num=True)
    return labeled_image, count

debug_index=0
# for debug_index in range(13824):
ress=analyze_all_control_points(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points
                               ,pmapped_batch_size,sv_diameter
                               ,r,diam_x,diam_y,half_r)
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
_,count0=connected_components(ress[0,:,:,0])
_,count1=connected_components(ress[0,:,:,1])
_,count2=connected_components(ress[0,:,:,2])
_,count3=connected_components(ress[0,:,:,3])

sns.heatmap(mask0,ax=axs[0,0]).legend([],[], frameon=False)
monox =-13
monoy =-5
# monox =(r//2)+1
# monoy =(r//2)+1
nexttx=(2*r)
nextty=(2*r)
plt.axhline(y=monox) 
plt.axvline(x=monoy) 
plt.axhline(y=monox+nexttx) 
plt.axvline(x=monoy+nextty) 
plt.axhline(y=monox+nexttx*2) 
plt.axvline(x=monoy+nextty*2) 
plt.axhline(y=monox+nexttx*3) 
plt.axvline(x=monoy+nextty*3) 
plt.axhline(y=monox+nexttx*4) 
plt.axvline(x=monoy+nextty*4) 

sns.heatmap(mask1,ax=axs[0,1]).legend([],[], frameon=False)
sns.heatmap(ress[0,:,:,2],ax=axs[1,0]).legend([],[], frameon=False)
sns.heatmap(ress[0,:,:,3],ax=axs[1,1]).legend([],[], frameon=False)

sum_count= count0+count1+count2+count3
# os.makedirs(f"/workspaces/jax_cpu_experiments_b/explore/debuggin_reshuffle_channels/{}" ,exist_ok = True)

# file_name=f"/workspaces/jax_cpu_experiments_b/explore/debuggin_reshuffle_channels/example_mask{debug_index}.png"
file_name=f"/workspaces/jax_cpu_experiments_b/explore/all_masks.png"
plt.savefig(file_name)
plt.clf()


# print(f"debug_index {debug_index} count {sum_count}  minn {jnp.min(ress.flatten())} maxx {jnp.max(ress.flatten())} ")
# # plt.show()
# plt.clf()

diam_x=32 #256+r
diam_y=32 #256+r
img_size=(1,diam_x,diam_y)
orig_grid_shape=grid_a_points.shape

sh_re_consts=get_simple_sh_resh_consts(img_size,r)
disp= list(map(lambda i: reshape_to_svs(ress,sh_re_consts[i],i) ,range(4)))
disp= jnp.concatenate(disp,axis=1)

for i in range(disp.shape[1]):
    sns.heatmap(disp[0,i,:,:])
    path= f"/workspaces/jax_cpu_experiments_b/explore/debuggin_reshuffle_channels/subm{i}.png"
    plt.savefig(path)
    plt.clf()
#     # plt.show()
    





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



