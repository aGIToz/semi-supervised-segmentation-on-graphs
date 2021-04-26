__kernel void compute_weights(__global float *dists_gpu, __global float *weight_vec, const int k, const float scale)
{
    int nk, i, j;
    nk = get_global_size(0);
    i = get_global_id(0);

    weight_vec[i] =  exp((-1.0f)*(dists_gpu[i]*dists_gpu[i]) / (scale) ); 
}
