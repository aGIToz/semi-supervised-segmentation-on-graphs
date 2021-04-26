__kernel void laplacian_filter(__global float *gray_old, __global float *gray, __global float *new_gray, __global int *ngbrs , __global float *wgts, const int k, const float lamb, const float pl, const float epsilon, const int chnl)

{ 
	int n, i, j, c, c2;
	n = get_global_size(0);
	i = get_global_id(0);
	float plhalf = pl / 2.0f;
	float plmin2 = (pl - 2.0f) / 2.0f;
	float tmp, tmp2, tmp3, sm;
 	/*<for harcoded k the code is very easy> the forward euler method comes here */
	for (c = 0; c < chnl; ++c)
	{
	tmp = 0.0; tmp2 = 0.0; tmp3 = 0.0;
	for (j = 0; j < k;)
	{
		sm = 0.0;
		for (c2 = 0; c2 < chnl; ++c2)
		{
		// sm += pown((gray[(n*c2) + ngbrs[i * k + j]] - gray[(n*c2)+i]),2);
		sm += ((gray[(n*c2) + ngbrs[i * k + j]] - gray[(n*c2)+i])*(gray[(n*c2) + ngbrs[i * k + j]] - gray[(n*c2)+i]));
		}
		tmp +=	pow(wgts[i*k + j],plhalf) * pow((sm + epsilon),(plmin2)) * gray[(n*c) + ngbrs[i*k+j]];
		tmp2 += pow(wgts[i*k + j],plhalf) * (pow((sm + epsilon),(plmin2)));
		j = j + 1;
	} 
	new_gray[(n*c) + i] =  (tmp + (lamb*gray_old[(n*c)+i]) )/ (tmp2 + lamb );
	}

 }
