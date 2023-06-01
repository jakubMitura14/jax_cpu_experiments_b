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
from simple_seg_utils import *
from simple_seg_get_edges import *
from GNN_model import *
from simple_graph_segmentation import *
import tensorflow as tf

config.update("jax_debug_nans", True)
cfg=get_cfg()
edge_pairs=get_sorce_targets(cfg.orig_grid_shape)
model= Simple_graph_net(cfg,edge_pairs)


def setup_tensorboard():
    jax.numpy.set_printoptions(linewidth=400)
    ##### tensor board
    #just removing to reduce memory usage of tensorboard logs
    tensorboard_dir='/workspaces/jax_cpu_experiments_b/explore/tensorboard'
    shutil.rmtree(tensorboard_dir)
    os.makedirs(tensorboard_dir)


    # initialise_tracking()

    logdir=tensorboard_dir
    # plt.rcParams["savefig.bbox"] = 'tight'
    file_writer = tf.summary.create_file_writer(logdir)
    return file_writer

file_writer=setup_tensorboard()



# @functools.partial(jax.pmap,static_broadcasted_argnums=(1,2,3), axis_name='ensemble')#,static_broadcasted_argnums=(2)
def initt(rng_2,cfg:ml_collections.config_dict.FrozenConfigDict,model):
  """Creates initial `TrainState`."""
  img_size=list(cfg.img_size)
  masks_size=(cfg.img_size[0],cfg.img_size[1],cfg.img_size[2],8)  
  lab_size=list(cfg.label_size)
  # img_size[0]=img_size[0]//jax.local_device_count()
  # lab_size[0]=lab_size[0]//jax.local_device_count()
  input=jnp.ones(tuple(img_size))
  input_label=jnp.ones(tuple(lab_size))
  input_masks=jnp.ones(masks_size)
  rng_main,rng_mean=jax.random.split(rng_2)

  #jax.random.split(rng_2,num=1 )
  params = model.init({'params': rng_main,'to_shuffle':rng_mean  }, input,input_label,input_masks)['params'] # initialize parameters by passing a template image #,'texture' : rng_mean
  # cosine_decay_scheduler = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.total_steps, alpha=0.95)#,exponent=1.1
  decay_scheduler=optax.linear_schedule(cfg.learning_rate, cfg.learning_rate/10, cfg.total_steps, transition_begin=0)
  
  tx = optax.chain(
        optax.clip_by_global_norm(3.0),  # Clip gradients at norm 
        # optax.lion(learning_rate=cfg.learning_rate)
        optax.lion(learning_rate=decay_scheduler)
        # optax.adafactor()
        
        )

  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4))
# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4))
# @partial(jax.jit,static_argnames=("model"))
@jax.jit
def update_fn(state, image, label,masks):
  """Train for a single step."""
  def loss_fn(params,image,label,masks):
    losses,preds,orig_label_sv=model.apply({'params': params}, image,label,masks)#, rngs={'texture': random.PRNGKey(2)}
    return jnp.mean(losses) 

  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,label,masks)
  state=state.apply_gradients(grads=grads)

  return state,l



# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4,5))
def simple_apply(state, image, labels,masks,cfg,step,model):
  losses,preds,orig_label_sv=model.apply({'params': state.params}, image,labels,masks, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}


  return losses,preds,orig_label_sv

def get_visualizations(masks,batch_images,label_preds,epoch,tt):
  # print(f"aaa label_preds {jnp.sum(label_preds.flatten())} shape {label_preds.shape}")
  # label_preds=nn.softmax(label_preds,axis=-1)
  # label_preds= jnp.round(label_preds)
  # print(f"bbb label_preds {jnp.sum(label_preds.flatten())} label_preds {label_preds.shape} ")
  label_preds=label_preds[:,1]
  print(f"ccc label_preds {jnp.sum(label_preds.flatten())}")

  initial_masks= jnp.stack([
      get_initial_supervoxel_masks(cfg.orig_grid_shape,0,0),
      get_initial_supervoxel_masks(cfg.orig_grid_shape,1,0),
      get_initial_supervoxel_masks(cfg.orig_grid_shape,0,1),
      get_initial_supervoxel_masks(cfg.orig_grid_shape,1,1)
          ],axis=0)
  initial_masks=jnp.sum(initial_masks,axis=0)   
  initial_masks=einops.rearrange(initial_masks,'x y p ->1 x y p')
  initial_masks=initial_masks
  shape_reshape_cfgs=get_all_shape_reshape_constants(cfg,r_x=3,r_y=3)
  shape_reshape_cfgs_old=get_all_shape_reshape_constants(cfg,r_x=3,r_y=2)

  tokens_a,tokens_b,tokens_c,tokens_d,tokens_a_shape, tokens_b_shape,tokens_c_shape,tokens_d_shape=iter_over_all_masks_for_shape(masks,batch_images,shape_reshape_cfgs,shape_reshape_cfgs_old,initial_masks,False,cfg.epsilon,True)
  # print(f"tokens_a_shape {tokens_a_shape} tokens_b_shape {tokens_b_shape} tokens_c_shape {tokens_c_shape} tokens_d_shape {tokens_d_shape}" )   
  
  recreated_dense_label=reshape_sv_labels_to_dense(tokens_a_shape, tokens_b_shape,tokens_c_shape,tokens_d_shape
                               ,tokens_a,tokens_b,tokens_c,tokens_d
                               ,label_preds,shape_reshape_cfgs
                               )
  with file_writer.as_default():
    tf.summary.image(f"inferred_labels {tt}",plot_heatmap_to_image(recreated_dense_label[0,:,:,0]) , step=epoch,max_outputs=2000)
    print(f"recreated_dense_label {jnp.sum(recreated_dense_label.flatten())} shape {recreated_dense_label.shape}")

  # print(f"recreated_dense_label {recreated_dense_label.shape}")



def train_epoch(batch_images,batch_labels,masks,epoch,index
                ,model,cfg
                ,rng_loop
                ,state
                ):    
  epoch_loss=[]
  # params_repl = flax.jax_utils.replicate(params_cpu)
  # opt_repl = flax.jax_utils.replicate(opt_cpu)
  rngs_loop = flax.jax_utils.replicate(rng_loop)
  # print(f"state {state[1]}")
  state,loss=update_fn(state, batch_images, batch_labels,masks)
  epoch_loss.append(jnp.mean(loss).flatten())

  # if(index==0 and epoch%cfg.divisor_logging==0):
  #   # losses,masks,out_image=model.apply({'params': state.params}, batch_images[0,:,:,:,:],dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
  #   losses,masks,out_image=simple_apply(state, batch_images, batch_labels,cfg,index,model)
  #   #overwriting masks each time and saving for some tests and debugging
  #   #saving images for monitoring ...
  #   with file_writer.as_default():
  #       tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)    

  #we need to keep those below with batch =1

  batch_images_for_vis=batch_images
  batch_labels_for_vis=batch_labels
  masks_for_vis=masks
  if (epoch%8==0):
    losses,preds,orig_label_sv=simple_apply(state, batch_images_for_vis, batch_labels_for_vis,masks_for_vis,cfg,step,model)
    get_visualizations(masks,batch_images,preds,epoch,'inferred')
    get_visualizations(masks,batch_images,orig_label_sv,epoch,'orig')

    with file_writer.as_default():
        tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
         
  return state,loss




def main_train():
  slicee=57#57 was nice

  prng = jax.random.PRNGKey(42)

  rng_2=jax.random.split(prng,num=jax.local_device_count() )


  f = h5py.File('/workspaces/jax_cpu_experiments_b/hdf5_loc/example_mask.hdf5', 'r+')
  label=f["label"][:,:]
  curr_image=f["image"][:,:]
  masks=f["masks"][:,:,:]
  label= einops.rearrange(label,'w h -> 1 w h 1')
  curr_image= einops.rearrange(curr_image,'w h -> 1 w h 1')
  masks= einops.rearrange(masks,'w h c -> 1 w h c')

  state= initt(prng,cfg,model)  

  print(f"sum orig label {jnp.sum(label.flatten())} shape {label.shape}")
  with file_writer.as_default():
    tf.summary.image(f"orig label",plot_heatmap_to_image(jnp.round(label)[0,:,:,0]) , step=1)  



  for epoch in range(1, cfg.total_steps):
      prng, rng_loop = jax.random.split(prng, 2)
      print(f"epoch {epoch} ")
      state,loss=train_epoch(curr_image,label,masks,epoch,0
                                        #,tx, sched_fns,params_cpu
                                        ,model,cfg
                                        #,opt_cpu,sched_fns_cpu
                                        ,rng_loop,
                                      #  ,params_repl, opt_repl
                                        state)
      print(f"loss {loss}")




tic_loop = time.perf_counter()

main_train()

x = random.uniform(random.PRNGKey(0), (100, 100))
jnp.dot(x, x).block_until_ready() 
toc_loop = time.perf_counter()
print(f"loop {toc_loop - tic_loop:0.4f} seconds")



# tensorboard --logdir=/workspaces/jax_cpu_experiments_b/explore/tensorboard
