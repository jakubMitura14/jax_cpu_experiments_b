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
  print(f"in initttt {tuple(img_size)}")
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

def update_fn(state, image, label,masks,cfg,model):
  """Train for a single step."""
  def loss_fn(params,image,label,masks):
    losses=model.apply({'params': params}, image,label,masks)#, rngs={'texture': random.PRNGKey(2)}
    return jnp.mean(losses) 

  grad_fn = jax.value_and_grad(loss_fn)
  l, grads = grad_fn(state.params,image,label,masks)
  state=state.apply_gradients(grads=grads)

  return state,l



# @partial(jax.pmap, axis_name="batch",static_broadcasted_argnums=(2,3,4,5))
def simple_apply(state, image, labels,masks,cfg,step,model):
  losses=model.apply({'params': state.params}, image,labels,masks, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}


  return losses



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
  state,loss=update_fn(state, batch_images, batch_labels,masks,cfg,model)
  epoch_loss.append(jnp.mean(loss).flatten())

  # if(index==0 and epoch%cfg.divisor_logging==0):
  #   # losses,masks,out_image=model.apply({'params': state.params}, batch_images[0,:,:,:,:],dynamic_cfg, rngs={'to_shuffle': random.PRNGKey(2)})#, rngs={'texture': random.PRNGKey(2)}
  #   losses,masks,out_image=simple_apply(state, batch_images, batch_labels,cfg,index,model)
  #   #overwriting masks each time and saving for some tests and debugging
  #   #saving images for monitoring ...
  #   with file_writer.as_default():
  #       tf.summary.scalar(f"mask_0 mean", np.mean(mask_0.flatten()), step=epoch)    

  with file_writer.as_default():
      tf.summary.scalar(f"train loss ", np.mean(epoch_loss),       step=epoch)
         
  return state,loss




def main_train(cfg):
  slicee=57#57 was nice

  prng = jax.random.PRNGKey(42)
  model= Simple_graph_net(cfg)
  rng_2=jax.random.split(prng,num=jax.local_device_count() )


  f = h5py.File('/workspaces/jax_cpu_experiments_b/hdf5_loc/example_mask.hdf5', 'r+')
  label=f["label"][:,:]
  curr_image=f["image"][:,:]
  masks=f["masks"][:,:,:]
  label= einops.rearrange(label,'w h -> 1 w h 1')
  curr_image= einops.rearrange(curr_image,'w h -> 1 w h 1')
  masks= einops.rearrange(masks,'w h c -> 1 w h c')



  state= initt(prng,cfg,model)  




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




tic_loop = time.perf_counter()

main_train(get_cfg())

x = random.uniform(random.PRNGKey(0), (100, 100))
jnp.dot(x, x).block_until_ready() 
toc_loop = time.perf_counter()
print(f"loop {toc_loop - tic_loop:0.4f} seconds")