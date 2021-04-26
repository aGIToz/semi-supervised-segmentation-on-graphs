"""
Iso-tropic filtering and compression as discussed with Julien Rabin.
For more check-out the Book.
Write down the latex equation.

todo:
OT = optimize
remove  the unessary print statements
swap is not correct for the filtering.
comments are not proper

done:
better variable names
"""

from time import time
import pyopencl as cl
import numpy as np
import argparse
import os
import sys
pwd = os.getcwd()
bool_1 = False
print(pwd) if bool_1 else print()
mywf = os.path.join(pwd,'filtering_iso/iso.cl')

class hyperParams:
        """ A basic structure to group all the hyper params
        params:
        ------
        lamb:
        it:
        ep:
        pl:

        """ 

        def __init__(self, lamb, it, ep, pl):
                self.lamb = lamb
                self.it = it
                self.ep = ep
                self.pl = pl

def isoFilter(graph, signal, hp):
        """Does the isoprotic filtering on the graph
        params
        ------
        graph: 
        signal:
        hp: hyperparameters

        returns
        -------
        new_signal:
        """
        ngbrs = graph.ngbrs
        wgts = graph.wgts
        k = graph.k
        ngbrs = ngbrs.astype('int32')
        wgts = wgts.astype('float32')
        n, chnl = signal.shape
        signal = np.reshape(signal,(n*chnl),order='F')
        signal = signal.astype('float32')
        print("signal",signal.shape) if bool_1 else print()
        signal_old = np.copy(signal) 
        print("n",n) if bool_1 else print()
        lamb = hp.lamb
        it = hp.it
        pl = hp.pl
        epsilon = hp.ep
        print("success till loading") if bool_1 else print()
        
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
        signal_old_buf= cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=signal_old)
        weight_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=wgts)

        #need to create new intensity buffers
        new_signal= np.ndarray(shape=(n*chnl,), dtype=np.float32)
        new_signal_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, new_signal.nbytes)

        #run the kernel here in a loop
        import time 
        start = time.time()
        for uv in range(0, it):
                program.laplacian_filter(queue, (n,), None, signal_old_buf, signal_buf, new_signal_buf, ngbrs_buf, weight_buf, np.int32(k), np.float32(lamb) ,np.float32(pl), np.float32(epsilon), np.int32(chnl))
                #swap
                signal_buf,new_signal_buf=new_signal_buf,signal_buf
        queue.finish()
        end = time.time() - start
        print(f"time taken is {end}")
        # copy the new signal vec
        cl.enqueue_copy(queue, new_signal, new_signal_buf)
        # save the new intensity vec here
        print("finish") if bool_1 else print()
        return np.reshape(new_signal,(int(len(new_signal)/chnl),chnl),order="F")
