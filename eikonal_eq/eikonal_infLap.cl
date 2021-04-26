// implementation for the infinity laplacian

__kernel void laplacian_filter( __global float *signal, __global float *new_signal,   __global int *ngbrs , __global float *wgts, const int k, const int chnl)

{ 
	int n, i, j,  r, c;
	n = get_global_size(0);
	i = get_global_id(0);
    float plhalf = 0.5f;
	float tmp;
	tmp = 0.0;
	float max_c;
	max_c = 0.0;

    for (c=0; c< chnl; ++c)
    {
	{
	/* computation of (grad^- f)^p*/
	for (j = 0; j < k; )
	{
		tmp = pow(wgts[i*k + j],plhalf) * (fmax((signal[(n*c) + i] - signal[(n*c) + ngbrs[i*k+j]]),0.0f));
        if ( j == 0 )
        {
            max_c = tmp;
        }

        if ( max_c < tmp)
        {
            max_c = tmp;
        }

		j = j + 1;
	}

	new_signal[(n*c) + i] = signal[(n*c) + i] + (1.0f * (1.0f - max_c));
	}
	
 }
 }
