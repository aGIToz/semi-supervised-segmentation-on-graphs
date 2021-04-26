// implementation for the infinity laplacian
int isin(__global int *, const int , const int );

__kernel void laplacian_filter( __global int *A, __global float *red, __global float *new_red,   __global int *ngbrs , __global float *wgts, const int k,  const float pl, const float dt, const int size_A, const int chnl)

{ 
	int n, i, j,  r, c;
	n = get_global_size(0);
	i = get_global_id(0);
    float plhalf = pl / 2.0f;
	//float plhalf = pl ;
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
		tmp = pow(wgts[i*k + j],plhalf) * (fmax((red[(n*c) + i] - red[(n*c) + ngbrs[i*k+j]]),0.0f));
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

	new_red[(n*c) + i] = red[(n*c) + i] + (dt * (1.0f - max_c));
	}
	
 }
 }
