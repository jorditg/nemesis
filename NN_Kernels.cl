/**********************************************************************
Copyright 2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define TILEX 4

#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2


float4 sigmoid(float4 x)
{
  return 1.0f / ( 1.0f + exp( -x ) ); 
}

float4 sigmoid_derivative(float4 sigmoid)
{
  return sigmoid*(sigmoid - 1.0f);
}

float4 cross_entropy(float4 y, float4 t)
{
  return ( t * ln(y) + (1.0f - t) * ln (1.0f - y) );
}

float4 cross_entropy_derivative(float4 y, float4 t)
{
  return ( ( t - y ) / ( y * (1.0f - y) ) );
}

/* Matrix A is cached into local memory block */
/* Required global threads = (colsC / 4, rowsC / 4) */
__kernel void matrixMultiplicationSigmoidKernelLocal
                             (__global float4 *matrixA,
                              __global float4 *matrixB,
                              __global float4* matrixC,
                              int widthA,
                              int offsetA,
                              int offsetB,
                              int offsetC,
                              __local float4 *blockA,
                              int setBias,
                              int calcSigmoid)
{
    int blockPos = get_local_id(0) + get_local_size(0) * (get_local_id(1) << TILEY_SHIFT); //Should be : localId * (TILEX / 4) (float4)
    
    /* Position of thread will be according to the number of values it writes i.e TILE size */
    int globalPos = offsetC/4 + get_global_id(0) + (get_global_id(1) << TILEY_SHIFT) * get_global_size(0);

    /* Each thread writes 4 float4s */
    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    int temp = widthA / 4;

    /* This loop runs for number of blocks of A in horizontal direction */
    for(int i = 0; i < (temp / get_local_size(0)); i++)
    {
        /* Calculate global ids of threads from the particular block to load from matrix A depending on i */
        int globalPosA = offsetA/4 + i * get_local_size(0) + get_local_id(0) + (get_global_id(1) << TILEY_SHIFT) * temp;

        /* Load values in blockA from matrixA */
        blockA[blockPos] = matrixA[globalPosA];
        blockA[blockPos + get_local_size(0)] = matrixA[globalPosA + temp];
        blockA[blockPos + 2 * get_local_size(0)] = matrixA[globalPosA + 2 * temp];
        blockA[blockPos + 3 * get_local_size(0)] = matrixA[globalPosA + 3 * temp];

        barrier(CLK_LOCAL_MEM_FENCE);

        /* Calculate global ids of threads from the particular block to load from matrix B depending on i */
        int globalPosB = offsetB/4 + get_global_id(0) + ((i * get_local_size(0)) << TILEY_SHIFT) * get_global_size(0);

        /* This loop runs for number of threads in horizontal direction in the block of A */
        for(int j = 0; j < get_local_size(0) * 4; j=j+4)
        {
            /* Load 4 float4s from blockA : access patters = strided from local memory */
            float4 tempA0 = blockA[(j >> 2) + get_local_id(1) * TILEY * get_local_size(0)];
            float4 tempA1 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 1) * get_local_size(0)];
            float4 tempA2 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 2) * get_local_size(0)];
            float4 tempA3 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 3) * get_local_size(0)];

            /* Load corresponding values from matrixB, access pattern = linear from global memory */
            float4 tempB0 = matrixB[globalPosB  + j *  get_global_size(0)]; //Should be localId.x * (TILEX / 4)
            float4 tempB1 = matrixB[globalPosB  + (j + 1) * get_global_size(0)];
            float4 tempB2 = matrixB[globalPosB  + (j + 2) * get_global_size(0)];
            float4 tempB3 = matrixB[globalPosB  + (j + 3) * get_global_size(0)];
    
            sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
            sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
            sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
            sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

            sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
            sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
            sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
            sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

            sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
            sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
            sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
            sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

            sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
            sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
            sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
            sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

        }
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Calculate the sigmoid function of the sum
    if(calcSigmoid) {
        sum0 = sigmoid(-sum0);
        sum1 = sigmoid(-sum1);
        sum2 = sigmoid(-sum2);
        sum3 = sigmoid(-sum3);
    }

    // The first neuron of every layer exept the last one is the BIAS NEURON, that is to say that it always has a 1.0 output.
    // setBias must be true for all the layers calculations except the last one (the output one).
    if(setBias && get_global_id(0) == 0) {
        sum0.x = 1.0f;
        sum1.x = 1.0f;
        sum2.x = 1.0f;
        sum3.x = 1.0f;
    }    

    // end of calculation of sigmoid function
    
    /* Write 16 values to matrixC */
    matrixC[globalPos] = sum0;
    matrixC[globalPos +  get_global_size(0)] = sum1;
    matrixC[globalPos +  2 * get_global_size(0)] = sum2;
    matrixC[globalPos +  3 * get_global_size(0)] = sum3;    
}

/* Substracts element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSubstractKernel(__global float4 *t,
                                         __global float4 *y,
                                         __global float4* delta)
{
    int i = get_global_id(0);
    delta[i] = t[i] - y[i];
}

__kernel void crossEntropyKernelLocal(__global float4* y, 
                                      __global float4 t, 
                                      __global float4* output, 
                                      __local float4* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    float4 y1 = y[stride];
	float4 t1 = t[stride];
	float4 i1 = cross_entropy(t1, y1);
	float4 y2 = y[stride + 1];
	float4 t2 = t[stride + 1]];
	float4 i2 = cross_entropy(t2, y2);
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