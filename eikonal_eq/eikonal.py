"""
Azad Amitoz 2021-04-26 14:27
Eikonal equation on weighted graphs.
Ref: Vinh Thong Ta, Abderrahim Elmoataz, Olivier Lézoray. Mathematical Morphology and Eikonal Equations on Graphs for Nonlocal Image and Data Processing. 2009. 
"""

from time import time
import pyopencl as cl
import numpy as np
import argparse
import os
import sys
# changes to the path are made here 
pwd = os.getcwd()
base_d = os.path.dirname(pwd)
bool_1 = False
print(base_d) if bool_1 else print()
mywf = os.path.join(pwd,'eikonal_eq/eikonal_infLap.cl')
util_d = os.path.join(base_d,'utils')
sys.path.append(util_d)


class hyperParams:
        """ A basic structure to group all the hyper params
        params:
        ------
        dt
        it:
        pl:

        """ 
        def __init__(self, dt, it, pl):
                self.dt = dt
                self.it = it
                self.pl = pl


def eikonal(graph, signal, hp):
        """Does the mean-curvature evolution 
        params:
        ------
        graph: 
        signal: A initial distance field, for m number of seeds
        it is of size (n X m=chnls).
        hp: hyperparameters

        return:
        ------
        new_signal:
        """

        ngbrs = graph.ngbrs
        wgts = graph.wgts
        k = graph.k
        ngbrs = ngbrs.astype('int32')
        wgts = wgts.astype('float32')
        n, chnl = signal.shape
        """
        old notes, need to include the A set here.
        red =  gray[:,0]
        # get the ids of the seed
        """
        A = np.where(~signal.any(axis=1))[0]
        A = A.astype('int32')
        size_A = len(A)
        signal = np.reshape(signal,(n*chnl),order='F')
        signal = signal.astype('float32')
        print("signal",signal.shape) if bool_1 else print()
        print("n",n) if bool_1 else print()
        dt = hp.dt
        it = hp.it
        pl = hp.pl
        print("sucess till loading") if bool_1 else print()

        # create the opencl context
        platform = cl.get_platforms()[0]
        print(platform)
        device = platform.get_devices()[0]
        print(device)
        context = cl.Context([device])
        print(context)
        program = cl.Program(context, open(mywf).read()).build()
        queue = cl.CommandQueue(context)
        print(queue)

        #create the buffers now.
        mem_flags = cl.mem_flags
        ngbrs_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,hostbuf=ngbrs) 
        signal_buf= cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=signal)
        A_buf= cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A)
        weight_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=wgts)

        #need to create new signal 
        new_signal= np.ndarray(shape=(n*chnl,), dtype=np.float32)
        new_signal_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, new_signal.nbytes)

        #run the kernel here in a loop
        for uv in range(0, it):
                program.laplacian_filter(queue, (n*chnl,), None, A_buf,  signal_buf, new_signal_buf,  ngbrs_buf, weight_buf, np.int32(k), np.float32(pl), np.float32(dt),  np.int32(size_A), np.int32(chnl))
                signal_buf,new_signal_buf=new_signal_buf,signal_buf

        # copy the new intensity vec
        cl.enqueue_copy(queue, new_signal, new_signal_buf)
        # save the new intensity vec here
        print("finish") if bool_1 else print()
        return np.reshape(new_signal,(int(len(new_signal)/chnl),chnl),order="F")
