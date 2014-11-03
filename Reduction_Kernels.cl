
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each block invocation of this kernel, reduces block of input array
 * to a single value and writes this value to output.
 * 
 * Each work-item loads its data from input array to shared memory of block. 
 * Reduction of each block is done in multiple passes. In first pass 
 * first half work-items are active and they update their values in shared memory 
 * by adding other half values in shared memory. In subsequent passes number 
 * of active threads are reduced to half and they keep updating their value with 
 * other half values of shared memory. 
 */

__kernel
void 
cross_entropy(__global float4* y, __global float4 t, __global float4* output, __local float4* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    float4 y1 = y[stride];
	float4 t1 = t[stride];
	float4 i1 = t1 * ln(y1) + (1 - t1) * ln(1 - y1);
	float4 y2 = y[stride + 1];
	float4 t2 = t[stride + 1]];
	float4 i2 = t2 * ln(y2) + (1 - t2) * ln(1 - y2);
	sdata[tid] = i1 + i2;

    barrier(CLK_LOCAL_MEM_FENCE);
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = - sdata[0];	// cross entropy is the negative sum
}

// Al finalizar la función se obtiene un vector de output de tamaño igual al número de grupos
// que hay que sumar, obteniendo el resultado final
