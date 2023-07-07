import numpy as np
import matplotlib.pyplot as plt
import toolz
import optax
import jax.numpy as jnp
import jax
from flax import linen as nn
import einops
import seaborn as sns


def differentiable_abs(x):
    """ 
    differentiable approximation of absolute value function
    """
    a=4.0
    return x*jnp.tanh(a*x)


def get_triangle_area(p_0,p_1,p_2):
    area = 0.5 * (p_0[0] * (p_1[1] - p_2[1]) + p_1[0] * (p_2[1] - p_0[1]) + p_2[0]
                  * (p_0[1] - p_1[1]))
    return differentiable_abs(area)

def is_point_in_triangle(test_point,sv_center,control_point_a,control_point_b):
    """ 
    basic idea is that if a point is inside the triangle and we will create 3 sub triangles inside 
    where the new point is the apex and the bases are the 3 edges of the tested triangles
    if the sum of the areas of the sub triangles is equal the area of tested triangle the point is most probably inside the tested triangle
    if the sum of the subtriangles areas is diffrent then area of tested triangle it is for sure not in the triangle
    tested triangle will be always build from 3 points where sv center is one of them and other 2 points are sv control points
    adapted from https://stackoverflow.com/questions/59597399/area-of-triangle-using-3-sets-of-coordinates
    added power and sigmoid to the end to make sure that if point is in the triangle it will be approximately 0 and otherwise approximately 1
    """
    main_triangle_area= get_triangle_area(sv_center,control_point_a,control_point_b)
    sub_a=get_triangle_area(test_point,control_point_a,control_point_b)
    sub_b=get_triangle_area(sv_center,test_point,control_point_b)
    sub_c=get_triangle_area(sv_center,control_point_a,test_point)

    subtriangles_area= sub_a+sub_b+sub_c
    area_diff=main_triangle_area-subtriangles_area
    area_diff=jnp.power(area_diff,2)
    return (nn.sigmoid(area_diff*5000)-0.5)*2



def get_triangles_data():
    """ 
    manually setting data about what triangles are present in each square 
    it is based on set of sv centers and control points - also comments are to the upper left square of the image in 
    /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    we will also use the id of nodes as in the image bottom right
    single triangle data will consist of 4 entries - first 3 will be verticies ids as in image and the last one the id of the sv that is owner of this node
        as in the alpha order 
    """
    return jnp.array([[0,8,1,0]#A
         ,[0,1,2,0]#B
         ,[0,2,3,1]#C
         ,[0,3,4,1]#D
         ,[0,4,5,2]#L
         ,[0,5,6,2]#K
         ,[0,6,7,3]#J
         ,[0,7,8,3]]#I
        )

def analyze_single_triangle(curried,triangle_dat):
    """ 
    given a point it is designed to be scanned over triangles
    """
    x_y,control_points_coords,res=curried
    is_in=is_point_in_triangle(x_y,control_points_coords[triangle_dat[0],:],control_points_coords[triangle_dat[1],:],control_points_coords[triangle_dat[3],:])
    return (x_y,control_points_coords,res.at[triangle_dat[3]].set(res[triangle_dat[3]]+is_in )),None


def analyze_single_point(x_y,triangles_data,control_points_coords):
    """ 
    analyze thepoints of sv area (apart from edges) by checking it against the triangles
    we will scan over those triangles and return array of length 4 that will indicate to which sv given point is attached
    x_y - array with 2 entries indicating x and y coordinates of currently analyzed point
    triangles_data - constants describing points and to which sv they belong
    control_points_coords - coordinates of the sv centers and control points in order as indicated at image 
        /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    """
    curried=x_y,control_points_coords,jnp.zeros(4)
    res,_= jax.lax.scan(analyze_single_triangle,curried, triangles_data)
    return res[2]

v_analyze_single_point=jax.vmap(analyze_single_point,in_axes=(0,None,None))
v_v_analyze_single_point=jax.vmap(v_analyze_single_point,in_axes=(0,None,None))

def analyze_point_linear(curr_point, control_point,channel_up,channel_down):
    """ 
    as we are analyzing bottom and right border as lines we are just intrested weather given point is up or 
        down the axis from control point
    curr_point - float representing the position in axis of intrest of the point currently analyzed
    control_point - float representing the position in axis of intrest of the point currently analyzed
    channel_up - the channel owned by the sv up the axis
    channel_down - the channel owned by the sv down the axis
    """
    #will give close to 1 if test point is maller than control
    is_test_smaller_than_control=nn.sigmoid( (control_point-curr_point)*100 )
    res=jnp.zeros(4)
    res=res.at[channel_up].set(1-is_test_smaller_than_control)
    res=res.at[channel_down].set(is_test_smaller_than_control)
    return res

v_analyze_point_linear= jax.vmap(analyze_point_linear,in_axes=(0,None,None,None))

def reshuffle_channels(res,sv_area_type):
    """ 
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    we get res as the input that has the channels organised as if it is first type this funtion is to fix it
    """
    def alfa():
        return res
        
    def beta():
        #0 1 2 3 -> 1 0 3 2
        return res[:,:, jnp.array([1,0,3,2])]
        
    def delta():
        #0 1 2 3 -> 3 2 1 0
        return res[:,:, jnp.array([3,2,1,0])]

    def gamma():
        #0 1 2 3 -> 2 3 0 1
        return res[:,:, jnp.array([2,3,0,1])]

    functions_list=[alfa,beta,delta,gamma]
    return jax.lax.switch(sv_area_type,functions_list)    


def analyze_square(control_points_coords,diameter
                   ,triangles_data,sv_area_type):
    """ 
    analyzing single square where each corner is created by sv center
    triangles_data- constants describing triangles specified in get_triangles_data function
    diameter - diameter of sv area - what is important the right and bottom edges will be treated separately as their ownership calculations do not require triangle analysis
    control_points_coords - data about location of control points where entries are organized as in the main image
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    """
    #get grid of points and apply analyze_single_point ignoring left and bottom borders for now
    grid=einops.rearrange(jnp.mgrid[0:diameter, 0:diameter],'c x y-> x y c')
    grid_right= grid[-1,:,1]
    grid_bottom= grid[:,-1,0]
    grid=grid[0:-1,0:-1,:]
    res=v_v_analyze_single_point(grid,triangles_data,control_points_coords)
    #analyze bottom and right border we assume that we are in square alpha so right border is between sv 1vs2 and bottom 2 vs 3
    right=v_analyze_point_linear(grid_right,control_points_coords[4,1],1,2)
    bottom=v_analyze_point_linear(grid_bottom,control_points_coords[6,0],2,3)
    #recreate full grid
    res= jnp.pad(res,((0,1),(0,1),(0,0)))
    res=res.at[-1,:,:].set(right)
    res=res.at[:,-1,:].set(bottom)
    #reshuffle the order of channels so they id of svs will be consistent between areas
    res=reshuffle_channels(res,sv_area_type)
    #return 4 channel array where each channel tell about given sv and weather this point is owned but that sv
    return res

v_analyze_square=jax.vmap(analyze_square,in_axes=(0,None,None,0) )
v_v_analyze_square=jax.vmap(v_analyze_square,in_axes=(0,None,None,0) )
v_v_v_analyze_square=jax.vmap(v_v_analyze_square,in_axes=(0,None,None,0) )

def analyze_all_control_points(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points,pmapped_batch_size,sv_diameter):
    #pad the grid control points so we can divide it basically we need to enlarge grid_a_points, grid_b_points_x and grid_b_points_y
    gridd_bigger=einops.rearrange(jnp.mgrid[-r:diam_x+2*r:r,-r:diam_y+2*r:r],'c x y-> x y c')-half_r
    grid_a_points_big=einops.rearrange(jnp.mgrid[0:diam_x+r:r, 0:diam_y+r:r],'c x y-> x y c')-half_r
    grid_b_points_x_big= (gridd_bigger+jnp.array([half_r,0.0]))[1:-2,1:-1,:]
    grid_b_points_y_big= (gridd_bigger+jnp.array([0,half_r]))[1:-1,1:-2,:]
    #set up for batch size
    grid_a_points_big=einops.repeat(grid_a_points_big, 'x y c->b x y c', b=pmapped_batch_size) 
    grid_b_points_x_big=einops.repeat(grid_b_points_x_big, 'x y c->b x y c', b=pmapped_batch_size) 
    grid_b_points_y_big=einops.repeat(grid_b_points_y_big, 'x y c->b x y c', b=pmapped_batch_size) 
    #resetting learned control point positions

    grid_a_points_big=grid_a_points_big.at[:,1:-1,1:-1,:].set(grid_a_points)
    grid_b_points_x_big=grid_b_points_x_big.at[:,:,1:-1,:].set(grid_b_points_x)
    grid_b_points_y_big=grid_b_points_y_big.at[:,1:-1,:,:].set(grid_b_points_y)
    # #reshape control points so they can be used in vmap 
    # grid_b_points_x_big=einops.rearrange(grid_b_points_x_big,'x y c-> (x y) c')
    # grid_b_points_y_big=einops.rearrange(grid_b_points_y_big,'x y c-> (x y) c')
    # grid_c_points=einops.rearrange(grid_c_points,'x y c-> (x y) c')
    # grid_a_points_big=einops.rearrange(grid_a_points_big,'x y c-> (x y) c')
    #join control points and sv centers in control_points_coords in the order as indicated in image /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    control_points_coords=[
        grid_c_points#0
        ,grid_a_points_big[:,0:-1,0:-1,:]#1
        ,grid_b_points_x_big[:,:,0:-1,:]#2
        ,grid_a_points_big[:,1:,0:-1,:]#3
        ,grid_b_points_y_big[:,1:,:,:]#4
        ,grid_a_points_big[:,1:,1:,:]#5
        ,grid_b_points_x_big[:,:,1:,:]#6
        ,grid_a_points_big[:,0:-1,1:,:]#7
        ,grid_b_points_y_big[:,0:-1,:,:]#8
    ]
    control_points_coords=einops.rearrange(control_points_coords,'st b x y c ->b x y st c')
    #we prepare data for vmapping - sv area type has this checkerboard organization
    triangles_data=get_triangles_data()
    shh= grid_c_points.shape
    sv_area_type=jnp.zeros((shh[1],shh[2]),dtype=int)
    sv_area_type.at[1::2,0::2].set(3)
    sv_area_type.at[1::2,1::2].set(2)
    sv_area_type.at[0::2,1::2].set(1)
    sv_area_type=einops.repeat(sv_area_type, 'x y->b x y', b=pmapped_batch_size)
    #vmap over analyze_square    
    res=v_v_v_analyze_square(control_points_coords,sv_diameter,triangles_data,sv_area_type)

    #select part that we are intrested in and subtract pad value from coordinates basically reverse padding 
    ## cutting
    # res=res[:,1:-1,1:-1,:,:,:]
    res= einops.rearrange(res,' b w h x y c-> b (w x) (h y) c')

    
    #returns the 4 channel mask that encode the ownership of the pixels for each supervoxel
    return res



r=8
half_r=r/2
diam_x=48+r
diam_y=48+r
gridd=einops.rearrange(jnp.mgrid[r:diam_x:r, r:diam_y:r],'c x y-> x y c')-half_r
gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r

grid_a_points=gridd
grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]
grid_b_points_x= (gridd_bigger+jnp.array([half_r,0.0]))[0:-1,1:-1,:]
grid_b_points_y= (gridd_bigger+jnp.array([0,half_r]))[1:-1,0:-1,:]



sv_diameter=r
pmapped_batch_size=1
grid_a_points=einops.repeat(grid_a_points, 'x y c->b x y c', b=pmapped_batch_size) 
grid_b_points_x=einops.repeat(grid_b_points_x, 'x y c->b x y c', b=pmapped_batch_size) 
grid_b_points_y=einops.repeat(grid_b_points_y, 'x y c->b x y c', b=pmapped_batch_size) 
grid_c_points=einops.repeat(grid_c_points, 'x y c->b x y c', b=pmapped_batch_size) 

ress=analyze_all_control_points(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points,pmapped_batch_size,sv_diameter)
print(ress.shape)

mask0=ress[0,:,:,0]
sns.heatmap(mask0)
plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask0.png')
plt.show()
mask1=ress[0,:,:,1]
sns.heatmap(mask0)
plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask1.png')
