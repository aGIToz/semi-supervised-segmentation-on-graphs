"""
A simple knn graph creation with exponential weights.

todo:
python decorator for elapsed time?
Do if for patches.
Graph Symmetrization.
"""

from time import time
import numpy
from bufferkdtree.neighbors import NearestNeighbors
from utilities import roll, unroll
import pyopencl as cl
import numpy as np
import os
import sys
pwd = os.getcwd()
base_d = os.path.dirname(pwd)
bool_1 = False
print(base_d) if bool_1 else print()
print(pwd) if bool_1 else print()
mywf = os.path.join(pwd,"graph/expo.cl")

class initialParams:
        """ A basic structure to group all intial params for a graph creation.
        params:
        ------
        position: This can also be patches !
        signal: can be a texture or patches in case of some images
        k:
        sigma:
        """ 

        def __init__(self, position, signal, k, sigma):
                """
                >>>>NOTE<<<<: azad mar. 04 févr. 2020 10:31:52 CET
                Often the signal used to calculate the weights if not 
                the same as signal used in evolving a pde which can be
                often be the initial distance seeds, or the rgb values
                or the xyz values.
                """
                
                """
                {{{OPTIMIZATION}}}: azad mar. 04 févr. 2020 10:36:29 CET
                The code can be more faster if one is creating the patch
                based graph as buffer KD tree already calculates the euc-
                lidean distance, right now you are calculating the euclidean
                distance in the kernel.
                """
                
                self.position = position
                self.signal = signal 
                self.k = k
                self.sigma = sigma

class Graph:
        """ A graph data structure which shall be returned 
        params:
        ------
        wgts:
        ngbrs:
        k:
        """ 

        def __init__(self, wgts, ngbrs, k):
                self.wgts =wgts 
                self.ngbrs = ngbrs
                self.k = k

def buildGraph(ip, dev=0):
        """Builds the knn grap with intial params.
        params:
        ------
        ip: initial params

        return: 
        ------
        graph: graph object of Graph 
        """

        start = time()
        nbrs = NearestNeighbors(n_neighbors = ip.k + 1, algorithm="buffer_kd_tree", tree_depth=9, plat_dev_ids={0:[0]})    
        nbrs.fit(ip.position)
        dists, inds = nbrs.kneighbors(ip.position)
        print("success") if bool_1 else print()

        # now build the graph using those nns using gpu
        platform = cl.get_platforms()[0]
        print(platform)
        device = platform.get_devices()[dev]
        print(device)
        context = cl.Context([device])
        print(context)
        program = cl.Program(context, open(mywf).read()).build()
        print(program)
        queue = cl.CommandQueue(context)
        print(queue)
        
         # define the input here which is the ndbrs gpu
        ngbrs_gpu = inds
        ngbrs_gpu = ngbrs_gpu[0:,1:]
        ngbrs_gpu = unroll(ngbrs_gpu)
        ngbrs_gpu = ngbrs_gpu.astype('int32')
        
         # define the second input here which is the signal levels
        signal =  ip.signal
        n, chnl = signal.shape
        signal = np.reshape(signal,(n*chnl,),order='F')
        signal = signal.astype('float32')
        print("signal",signal.shape) if bool_1 else print()
        k = ip.k
        print("n is :", n) if bool_1 else print()
        scale = ip.sigma
        
         # create the buffers on the device, intensity, nbgrs, weights
        mem_flags = cl.mem_flags
        ngbrs_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,hostbuf=ngbrs_gpu)
        signal_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=signal)
        weight_vec = np.ndarray(shape=(n*k,), dtype=np.float32)
        weight_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, weight_vec.nbytes)
        
         # run the kernel to compute the weights
        program.compute_weights(queue, (n,), None, signal_buf,  ngbrs_buf, weight_buf, np.int32(k), np.float32(scale), np.int32(chnl))
        
        queue.finish() #OT
        
        # copy the weihts to the host memory
        cl.enqueue_copy(queue, weight_vec, weight_buf)
        end = time() - start
        
        print('total time taken by the gpu python:', end) if bool_1 else print()
        # save the graph
        graph = Graph(weight_vec,ngbrs_gpu,k)
        return graph
