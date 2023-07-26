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
from .shape_reshape_functions import *
from functools import partial
import math

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
    return 1-(nn.sigmoid(area_diff*15)-0.5)*2



def lineLineIntersection(A, B, C, D):
    """ 
    based on https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    """
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
 
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
    

    determinant = (a1*b2 - a2*b1)+0.000000000001
    
    # if (determinant == 0):
    #     # The lines are parallel. This is simplified
    #     # by returning a pair of FLT_MAX
    #     return Point(10**9, 10**9)
    # else:
    x = (b2*c1 - b1*c2)/determinant
    y = (a1*c2 - a2*c1)/determinant
    return jnp.array([x, y])

def orthoProjection(vertex_a,vertex_b,vertex_c):
    """ 
    projection of a point on a line 
    adapted from https://stackoverflow.com/questions/55230528/find-point-where-altitude-meets-base-python
    """
    # abx = bx - ax
    abx = vertex_b[0] - vertex_a[0]
    # aby = by - ay
    aby = vertex_b[1] - vertex_a[1]
    # acx = cx - ax
    acx = vertex_c[0] - vertex_a[0]
    # acy = cy - ay
    acy = vertex_c[1] - vertex_a[1]
    t = (abx * acx + aby * acy) / ((abx * abx + aby * aby)+0.000000001)
    # px = ax + t * abx
    px = vertex_a[0] + t * abx
    # py = ay + t * aby
    py = vertex_a[1] + t * aby
    return jnp.array([px, py])


def get_point_on_a_line_b(vertex_0,vertex_1,weight):
    diff_x=vertex_1[0]-vertex_0[0]
    diff_y=vertex_1[1]-vertex_0[1]
    # weight=weight/2+1.5
    # weight=(weight*2)

    # C=jnp.array([vertex_0[0]+(diff_x*weight), vertex_0[1]])
    # D=jnp.array([vertex_0[0],vertex_0[1]+(diff_y*weight)])

    # alpha=np.pi/4
    # beta=-np.pi/4
    # a= vertex_0
    # #just multiplied by rotation matrix, we divide by square root of 2 in order to keep the weights in domain 1 to two
    # C= np.array([  a[0]*np.cos(alpha) + a[1]*np.sin(alpha), -a[0]*np.sin(alpha) + a[1]*np.cos(alpha)])/np.sqrt(2)
    # D= np.array([  a[0]*np.cos(beta) + a[1]*np.sin(beta), -a[0]*np.sin(beta) + a[1]*np.cos(beta)])/np.sqrt(2)
    
    # C=C*weight
    # D=D*weight
    # print(f"C {C} D {D} diff_x {diff_x} diff_y {diff_y}  vertex_0[0]{vertex_0[0]} diff_x*weight {diff_x*weight} sum {vertex_0[0]+(diff_x*weight)}")
    # return lineLineIntersection(vertex_0, vertex_1, C, D)
    return np.array([vertex_0[0]+(diff_x*weight),vertex_0[1]+(diff_y*weight)])




""" 
we want to put a new point in a triangle - that will be a new control point
point is as specified constrained by a triangle weights live on two of the primary triangle edges
so we take  2 edges establish positions of temporary points by moving on those edges by percentege of their length
then we get a line between those new points and apply 3rd weight to it so we will move along this new line
"""
def get_point_inside_triange(vertex_a,vertex_b,vertex_c,edge_weights):
    p0=get_point_on_a_line_b(vertex_a,vertex_b,edge_weights[0])
    p1=get_point_on_a_line_b(vertex_b,vertex_c,edge_weights[1])

    res=get_point_on_a_line_b(p0,p1,edge_weights[2])
    return res

def get_triangles_data():
    """ 
    manually setting data about what triangles are present in each square 
    it is based on set of sv centers and control points - also comments are to the upper left square of the image in 
    /workspaces/jax_cpu_experiments_b/geometric_sv_idea/triangle_geometric_sv.jpg
    we will also use the id of nodes as in the image bottom right
    single triangle data will consist of 4 entries - first 3 will be verticies ids as in image and the last one the id of the sv that is owner of this node
        as in the alpha order 
    """
    return jnp.array([
          [[0,8,7,3]#I
         ,[0,8,1,0]]#A

         ,[[0,2,1,0]#B
         ,[0,2,3,1]]#C

         ,[[0,4,3,1]#D
         ,[0,4,5,2]]#L

         ,[[0,6,5,2]#K
         ,[0,6,7,3]]]#J
        )


def analyze_single_triangle(curried,triangle_dat):
    """ 
    given a point it is designed to be scanned over triangles as we can add also additional
    control points 
    """
    x_y,control_points_coords,res=curried
    is_in=is_point_in_triangle(x_y,control_points_coords[triangle_dat[0],:],control_points_coords[triangle_dat[1],:],control_points_coords[triangle_dat[-1],:])
    krowa take into account changed triangles data and new triangles
    return (x_y,control_points_coords,res.at[triangle_dat[3]].set(res[triangle_dat[3]]+is_in )),None


def add_single_point(curried,added_points_index):
    main_triangle_dat,modified_control_points_coords,edge_weights,edge_weights_offset,adding_points_offset=curried
    new_edge_weights_offset=edge_weights_offset+3
    edge_weights_inner=edge_weights[edge_weights_offset:new_edge_weights_offset]
    vertex_a=modified_control_points_coords[main_triangle_dat[0,-4],:]# in first new point it would be yellow
    vertex_b=modified_control_points_coords[main_triangle_dat[0,-2],:]
    vertex_c=modified_control_points_coords[main_triangle_dat[0,-2],:]
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
    curried,new_points=jax.lax.scan(add_single_point,curried,jnp.arange(8,cfg.num_additional_points))
    main_triangle_dat,modified_control_points_coords,edge_weights,_,adding_points_offset=curried
    return main_triangle_dat,new_points

v_add_new_points_per_main_triangle= jax.vmap(add_new_points_per_main_triangle,in_axes=(0,None,None,None,None) )

def add_new_points(triangles_data,modified_control_points_coords,edge_weights,cfg):
    """ 
    after! we had applied learned weights to the primary control points we can add more by 
    adding between gird points b and c (yellow and green)  so in primary triangles data we 
    are inserting new point between entry 0 and 1 (second) of both subtriangles of main triangle
    (here example of main traingle is BC or DL with sub triangles B,C and D,L)
    """
    main_triangle_dats,new_points=v_add_new_points_per_main_triangle(jnp.arange(4),triangles_data,modified_control_points_coords,edge_weights,cfg)
    #we integrated modified triangle data from all primary triangles
    triangles_data=jnp.stack(main_triangle_dats,axis=0)
    new_points= einops.rearrange(new_points,'a b p-> (a b) p')
    modified_control_points_coords= jnp.concatenate([modified_control_points_coords,new_points],axis=0)
    return modified_control_points_coords,triangles_data


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
    return (res[2]+0.000000000000000000000000000000001)/(jnp.sum(res[2])+0.000000000000000000000000000000001)

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
    is_test_smaller_than_control=nn.sigmoid( (control_point-curr_point)*10 )
    res=jnp.zeros(4)
    res=res.at[channel_up].set(is_test_smaller_than_control)
    res=res.at[channel_down].set(1-is_test_smaller_than_control)
    return res

v_analyze_point_linear= jax.vmap(analyze_point_linear,in_axes=(0,None,None,None))

def reshuffle_channels(res,sv_area_type,debug_index):
    """ 
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    we get res as the input that has the channels organised as if it is first type this funtion is to fix it
    """
    # p = permutations([0,1,2,3])
    # p=list(p)
    # prod=list(product(p,p,p))
    
    def alfa():
        return res
        
    def beta():
        #0 1 2 3 -> 1 0 3 2
        return res[:,:, jnp.array([3, 2, 1, 0])]
        # return res[:,:, jnp.array([1,0,3,2])]
        return res
        
    def delta():
        #0 1 2 3 -> 3 2 1 0
        return res[:,:, jnp.array([2, 3, 0, 1])]   
        # return res[:,:, jnp.array([2,1,0,3])]
        # return res

    def gamma():
        #0 1 2 3 -> 2 3 0 1
        # return res[:,:, jnp.array([2,3,0,1])]
        return res[:,:, jnp.array([1, 0, 3, 2])]
        return res


    functions_list=[alfa,beta,delta,gamma]
    return jax.lax.switch(sv_area_type,functions_list)    




def analyze_square(control_points_coords,diameter
                   ,triangles_data,sv_area_type,debug_index):
    """ 
    analyzing single square where each corner is created by sv center
    triangles_data- constants describing triangles specified in get_triangles_data function
    diameter - diameter of sv area - what is important the right and bottom edges will be treated separately as their ownership calculations do not require triangle analysis
    control_points_coords - data about location of control points where entries are organized as in the main image
    sv_area_type - we have 4 sv_area_type in first node 0 is in upper left corner and we go clockwise so bottom left is 3
        other sv types are set in a way to be consistent with first sv_area_type
    """
    #get grid of points and apply analyze_single_point ignoring left and bottom borders for now
    grid=einops.rearrange(jnp.mgrid[0:diameter+1, 0:diameter+1],'c x y-> x y c')+control_points_coords[1,:]
    grid=grid[1:,1:,:]
    # print(f"\n ggggrid  \n {grid} \n  ")
    # print(f"grid {grid}")

    grid_right= grid[-1,:,1]
    grid_bottom= grid[:,-1,0]
    grid=grid[0:-1,0:-1,:]

    # print(f"grid {grid}")
    # example=analyze_single_point(grid[1,1],triangles_data,control_points_coords )
    # # print(f"eeeexample {example} \n point {grid[1,1]} \n triangles_data \n {triangles_data} \n control_points_coords \n {control_points_coords} \n")
    # x_y=grid[1,2]
    # print(f"x_y {x_y}")
    # for triangle_dat in triangles_data:
    #     is_in = is_point_in_triangle(x_y,control_points_coords[triangle_dat[0],:],control_points_coords[triangle_dat[1],:],control_points_coords[triangle_dat[2],:])
    #     print(f"is_in {is_in} \n triangle_dat {triangle_dat} \n p1 {control_points_coords[triangle_dat[0],:]} \n p2  {control_points_coords[triangle_dat[1],:]} p3 {control_points_coords[triangle_dat[2],:]}")

    res=v_v_analyze_single_point(grid,triangles_data,control_points_coords)
    #analyze bottom and right border we assume that we are in square alpha so right border is between sv 1vs2 and bottom 2 vs 3
    right=v_analyze_point_linear(grid_right,control_points_coords[4,1],1,2)
    bottom=v_analyze_point_linear(grid_bottom,control_points_coords[6,0],3,2)

    # right=right.at[0,:].set(0)
    right=right.at[-1,:].set(0)
    # bottom=bottom.at[0,:].set(0)
    bottom=bottom.at[-1,:].set(0)

    # right=right.at[0,1].set(1)
    right=right.at[-1,2].set(1)

    # bottom=bottom.at[0,3].set(1)
    bottom=bottom.at[-1,2].set(1)

    # recreate full grid
    res= jnp.pad(res,((0,1),(0,1),(0,0)))
    res=res.at[-1,:,:].set(right)
    res=res.at[:,-1,:].set(bottom)


    res=reshuffle_channels(res,sv_area_type,debug_index) 

    #return 4 channel array where each channel tell about given sv and weather this point is owned but that sv
    return res

v_analyze_square=jax.vmap(analyze_square,in_axes=(0,None,None,0,None) )
v_v_analyze_square=jax.vmap(v_analyze_square,in_axes=(0,None,None,0,None) )
v_v_v_analyze_square=jax.vmap(v_v_analyze_square,in_axes=(0,None,None,0,None) )

@partial(jax.jit, static_argnames=['pmapped_batch_size','sv_diameter','r','diam_x','diam_y','half_r'])
def analyze_all_control_points(grid_a_points,grid_b_points_x,grid_b_points_y,grid_c_points
                               ,pmapped_batch_size,sv_diameter
                               ,r,diam_x,diam_y,half_r):
    
    debug_index=0
    
    # sv_diameter_orig=sv_diameter
    # sv_diameter=sv_diameter-1
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
    sv_area_type=sv_area_type.at[1::2,0::2].set(3)
    sv_area_type=sv_area_type.at[1::2,1::2].set(2)
    sv_area_type=sv_area_type.at[0::2,1::2].set(1)
    sv_area_type=einops.repeat(sv_area_type, 'x y->b x y', b=pmapped_batch_size)

    # print(f"sv_area_type \n {sv_area_type[0,:,:]}")

    # example_coords=control_points_coords[0,1,2,:,:]
    # print(f"control_points_coords {example_coords}")
    # analyzed_example=analyze_square(example_coords ,sv_diameter,triangles_data,0)
    # print(f"analyzed_example_coords \n {analyzed_example.shape} \n {analyzed_example[:,:,0]} ")


    #vmap over analyze_square   
    # print(f"vvv control_points_coords {control_points_coords.shape} sv_diameter {sv_diameter} triangles_data {triangles_data.shape}") 
    res=v_v_v_analyze_square(control_points_coords,sv_diameter,triangles_data,sv_area_type,debug_index)


    # select part that we are intrested in and subtract pad value from coordinates basically reverse padding 
    # cutting
    # res=res[:,1:-1,1:-1,:,:,:]

    res= einops.rearrange(res,' b w h x y c-> b (w x) (h y) c')   
    print(f"tttt res {res.shape} half_r {half_r} ")
    half_r=int(half_r)
    res=res[:,half_r:-half_r,half_r:-half_r,:]
    # grid_c_points_f= einops.rearrange(grid_c_points[0,:,:,:],'x y c->(x y) c')
    # for coord in grid_c_points_f: #TODO remove
    #     print(coord)
    #     res=res.at[:,int(coord[0])+4,int(coord[1])+4,:].set(2)

    #returns the 4 channel mask that encode the ownership of the pixels for each supervoxel
    return res



r=8
half_r=r/2
diam_x=64 +r#256+r
diam_y=64 +r#256+r
gridd=einops.rearrange(jnp.mgrid[r:diam_x:r, r:diam_y:r],'c x y-> x y c')-half_r
gridd_bigger=einops.rearrange(jnp.mgrid[0:diam_x+r:r,0:diam_y+r:r],'c x y-> x y c')-half_r

grid_a_points=gridd
grid_c_points=(gridd_bigger+jnp.array([half_r,half_r]))[0:-1,0:-1,:]
grid_b_points_x= (gridd_bigger+jnp.array([half_r,0.0]))[0:-1,1:-1,:]
grid_b_points_y= (gridd_bigger+jnp.array([0,half_r]))[1:-1,0:-1,:]


# def add_grid_points_plus(mini_r,grid_c_points):
#     """ 
#     so we add the additional grid points between green and yellow ones 
#         at a distance mini_r from yellow

#     """
#     a= jnp.stack([grid_c_points[:,:,0]+mini_r,grid_c_points[:,:,1]+mini_r],axis=-1)
#     b= jnp.stack([grid_c_points[:,:,0]-mini_r,grid_c_points[:,:,1]+mini_r],axis=-1)
#     c= jnp.stack([grid_c_points[:,:,0]+mini_r,grid_c_points[:,:,1]-mini_r],axis=-1)
#     d= jnp.stack([grid_c_points[:,:,0]-mini_r,grid_c_points[:,:,1]-mini_r],axis=-1)
#     return (a,b,c,d)


# print(f"aaaa grid_a_points {grid_a_points.shape}")







######## random point mutation

# weights=np.ones((grid_a_points.shape[0],grid_a_points.shape[1],8))*2
# weights=np.ones_like(weights)#*110000000.0
# weights[2,2,:]=-10000.0

def get_contribution_in_axes(fixed_point,strength):
    # print(f"fixed_point {fixed_point} strength {strength}")
    e_x= jnp.array([1.0,0.0])
    e_y= jnp.array([0.0,1.0])
    x=optax.cosine_similarity(e_x,fixed_point)*strength
    y=optax.cosine_similarity(e_y,fixed_point)*strength
    return jnp.array([x,y])
v_get_contribution_in_axes=jax.vmap(get_contribution_in_axes)


def get_4_point_loc(points_const,point_weights,half_r):
    half_r_bigger=half_r#*1.2
    calced=v_get_contribution_in_axes(points_const,point_weights)

    
    calced=jnp.sum(calced,axis=0)
    # calced=calced/(jnp.max(calced.flatten())+0.00001)
    return calced*half_r_bigger

def divide_my(el):
    # print(f"aaaaaaaaaaaaa {el}")
    res=el[0]/jnp.sum(el)
    return jnp.array([res,1-res])
v_divide_my= jax.vmap( divide_my,in_axes=0)
v_v_divide_my= jax.vmap( v_divide_my,in_axes=0)

def get_b_x_weights(weights):
    weights_curr=weights[:,:,0:2] 
    grid_b_points_x_weights_0=np.pad(weights_curr[:,:,0],((1,0),(0,0)))
    grid_b_points_x_weights_1=np.pad(weights_curr[:,:,1],((0,1),(0,0)))
    grid_b_points_x_weights= np.stack([grid_b_points_x_weights_0,grid_b_points_x_weights_1],axis=-1)
    grid_b_points_x_weights=nn.sigmoid(grid_b_points_x_weights)
    return v_v_divide_my(grid_b_points_x_weights)


def get_b_y_weights(weights):
    weights_curr=weights[:,:,2:4] 
    grid_b_points_y_weights_0=np.pad(weights_curr[:,:,0],((0,0),(1,0)))
    grid_b_points_y_weights_1=np.pad(weights_curr[:,:,1],((0,0),(0,1)))
    grid_b_points_y_weights= np.stack([grid_b_points_y_weights_0,grid_b_points_y_weights_1],axis=-1)
    grid_b_points_y_weights=nn.sigmoid(grid_b_points_y_weights)
    return v_v_divide_my(grid_b_points_y_weights)
# return nn.softmax(grid_b_points_y_weights*100,axis=-1)




def get_for_four_weights(weights):
    """ 
        4- up_x,up_y
        5- up_x,down_y
        6- down_x,up_y
        7- down_x,down_y
    """
    up_x_up_y=np.pad(weights[:,:,4],((1,0),(1,0)))
    up_x_down_y=np.pad(weights[:,:,5],((1,0),(0,1)))
    down_x_up_y=np.pad(weights[:,:,6],((0,1),(1,0)))
    down_x_down_y=np.pad(weights[:,:,7],((0,1),(0,1)))

    grid_c_points_weights=np.stack([up_x_up_y,up_x_down_y,down_x_up_y,down_x_down_y],axis=-1)
    # print(f"grid_c_points_weights in get_for_four_weights {grid_c_points_weights} \n \n ")

    # print(f"grid_c_points {grid_c_points.shape} grid_c_points_weights {grid_c_points_weights.shape}")
    # return nn.tanh(grid_c_points_weights*100) 
    return nn.softmax(grid_c_points_weights*100,axis=-1) 

def apply_for_four_weights(grid_c_points_weight,grid_c_point,half_r):
    points_const=jnp.stack([  jnp.array([-half_r,-half_r])
                              ,jnp.array([-half_r,half_r])
                              ,jnp.array([half_r,-half_r])
                              ,jnp.array([half_r,half_r])
                              ],axis=0)

    calced=get_4_point_loc(points_const,grid_c_points_weight,half_r)

    return calced+grid_c_point
v_apply_for_four_weights=jax.vmap(apply_for_four_weights,in_axes=(0,0,None))
v_v_apply_for_four_weights=jax.vmap(v_apply_for_four_weights,in_axes=(0,0,None))


def move_in_axis(point,weights,axis,half_r ):
    """ 
    point can move up or down axis no more than half_r from current position 
    weights indicate how strongly it shoul go down (element 0) and up the axis  
    """
    return point.at[axis].set(point[axis]-weights[0]*half_r + weights[1]*half_r)
v_move_in_axis= jax.vmap(move_in_axis,in_axes=(0,0,None,None))
v_v_move_in_axis= jax.vmap(v_move_in_axis,in_axes=(0,0,None,None))



weights=(np.random.random((grid_a_points.shape[0],grid_a_points.shape[1],8))-0.5)*2
# weights=(np.random.random((grid_a_points.shape[0],grid_a_points.shape[1],8)))/2

# weights= jnp.ones_like(weights)
# weights=weights.at[1,1,:].set(3.0)


grid_b_points_x_weights=get_b_x_weights(weights)
grid_b_points_y_weights=get_b_y_weights(weights)

print(f"grid_b_points_x {grid_b_points_x.shape} grid_b_points_x_weights {grid_b_points_x_weights.shape} ")

grid_b_points_x=v_v_move_in_axis(grid_b_points_x,grid_b_points_x_weights,0, half_r)
grid_b_points_y=v_v_move_in_axis(grid_b_points_y,grid_b_points_y_weights,1, half_r)

grid_c_points_weights=get_for_four_weights(weights)
grid_c_points=v_v_apply_for_four_weights(grid_c_points_weights,grid_c_points,half_r)






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



# mask1=ress[0,:,:,2]
# sns.heatmap(mask1)
# plt.show()
# plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask2.png')

# mask1=ress[0,:,:,3]
# sns.heatmap(mask1)
# plt.show()
# plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask3.png')

# mask_sum=jnp.sum(ress[0,:,:,:],axis=-1)
# sns.heatmap(mask_sum)
# plt.savefig('/workspaces/jax_cpu_experiments_b/explore/example_mask_sum.png')
# plt.show()


# main grid_a_points (1, 32, 32, 2) grid_b_points_x (1, 33, 32, 2) grid_b_points_y (1, 32, 33, 2) grid_c_points (1, 33, 33, 2) pmapped_batch_size 1 sv_diameter 16 r 8 diam_x 264 diam_y 264 half_r 
# loca grid_a_points (1, 32, 32, 2) grid_b_points_x (1, 33, 32, 2) grid_b_points_y (1, 32, 33, 2) grid_c_points (1, 33, 33, 2) pmapped_batch_size 1 sv_diameter 16 r 8 diam_x 264 diam_y 264 half_r 


# vvv control_points_coords (1, 33, 33, 9, 2) sv_diameter 16 triangles_data (8, 4)
# 0 rrrrrrrrr (1, 33, 33, 16, 16, 4)


# python3 -m geometric_sv_idea.get_areas_from_control_points