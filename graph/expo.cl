__kernel void compute_weights(__global float *gray, __global int *ngbrs_gpu, __global float *weight_vec, const int k, const float scale, const int chnl)
{
    int n, i, j, c;
    n = get_global_size(0);
    i = get_global_id(0);
    float tmp;
    tmp = 0.0f;

    for (j = 0; j < k; ++j )
    {
		tmp = 0.0f;
		for (c = 0; c < chnl; ++c) 
		{
			tmp += pown((gray[(n*c) + i] - gray[(n * c) + ngbrs_gpu[i * k + j]]), 2); 
		}
               weight_vec[i*k+j] =  exp((-1.0f)*(tmp)/(scale * scale) ); 
   // printf("%s\n","this is a test string\n");            
   // printf("%f\n",red[i]);            
   // printf("%f\n",green[i]);            
               
    }

}


