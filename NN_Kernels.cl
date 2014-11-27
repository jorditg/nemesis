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

#include "NN_Kernels.h"

/*
 *  Returns the index of the element located in (row, col) in a 
 *  row-major memory ordered matrix (Fortran Type)
 */
int get_index(int offset, int r, int c, int nr_c)
{
    // TILEY_SHIFT due to we are indexing float4xfloat4 blocks of data
    return (offset + c + (r << TILEY_SHIFT)*nr_c);
}

float4 sigmoid(float4 x)
{
    return 1.0f / ( 1.0f + exp( -x ) ); 
}

float4 sigmoid_derivative(float4 sigmoid)
{
  return sigmoid*(sigmoid - 1.0f);
}

float4 cross_entropy(float4 t, float4 y)
{
    const float4 epsilon = 1E-30;
    return ( t * log(y + epsilon) + (1.0f - t) * log (1.0f - y + epsilon) );
}

/* Matrix A is cached into local memory block */
/* Required global threads = (colsC / 4, rowsC / 4) */
__kernel void matrixMultiplicationSigmoidKernelLocal
                             (__global float4 *matrixA,
                              __global float4 *matrixB,
                              __global float4 *matrixC,
                              int colsA,
                              int offsetA,
                              int offsetB,
                              int offsetC,
                              __local float4 *blockA,
                              int setBias,
                              int calcSigmoid,
                              int AInColMajorOrder,
                              int BInColMajorOrder,
                              int sumToMatrixC,
                              float multPrevVal,
                              float multSum)
{
    //int blockPos = get_local_id(0) + get_local_size(0) * (get_local_id(1) << TILEY_SHIFT); //Should be : localId * (TILEX / 4) (float4)
    int blockPos = get_index(0, get_local_size(0), get_local_id(0), get_local_id(1));

    /* Position of thread will be according to the number of values it writes i.e TILE size */
    
    //int globalPos = offsetC + get_global_id(0) + (get_global_id(1) << TILEY_SHIFT) * get_global_size(0);
    const int col_C = get_global_id(0);
    const int row_C = get_global_id(1);
    const int nr_cols_C = get_global_size(0);; 
    int globalPos = get_index(offsetC, row_C, col_C, nr_cols_C);

    /* Each thread writes 4 float4s */
    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    int temp = colsA / 4;
    
    /* This loop runs for number of blocks of A in horizontal direction */
    for(int i = 0; i < (temp / get_local_size(0)); i++)
    {
        /* Calculate global ids of threads from the particular block to load from matrix A depending on i */
        //int globalPosA = offsetA + i * get_local_size(0) + get_local_id(0) + (get_global_id(1) << TILEY_SHIFT) * temp;

        const int col_A = i * get_local_size(0) + get_local_id(0);
        const int row_A = get_global_id(1);
        const int nr_rows_A = get_global_size(1);
        const int nr_cols_A = temp; 
        
        if(!AInColMajorOrder) {
          int globalPosA = get_index(offsetA, row_A, col_A, nr_cols_A);
          /* Load values in blockA from matrixA */
          blockA[blockPos] = matrixA[globalPosA];
          blockA[blockPos + get_local_size(0)] = matrixA[globalPosA + nr_cols_A];
          blockA[blockPos + 2 * get_local_size(0)] = matrixA[globalPosA + 2 * nr_cols_A];
          blockA[blockPos + 3 * get_local_size(0)] = matrixA[globalPosA + 3 * nr_cols_A];
        } else {
          // If A is in column major order not only the index calculation is different but the float4xfloat4 block
          // of data has to be transposed
          int globalPosA = get_index(offsetA, col_A, row_A, nr_rows_A);
          // first of all we load the block to private memory
          float4 v1 = matrixA[globalPosA];
          float4 v2 = matrixA[globalPosA + 1 * nr_rows_A];
          float4 v3 = matrixA[globalPosA + 2 * nr_rows_A];
          float4 v4 = matrixA[globalPosA + 3 * nr_rows_A];

          // now we transpose it and assign it to the block of memory
          blockA[blockPos] = (float4) (v1.x, v2.x, v3.x, v4.x);
          blockA[blockPos + get_local_size(0)] = (float4) (v1.y, v2.y, v3.y, v4.y);
          blockA[blockPos + 2 * get_local_size(0)] = (float4) (v1.z, v2.z, v3.z, v4.z);
          blockA[blockPos + 3 * get_local_size(0)] = (float4) (v1.w, v2.w, v3.w, v4.w);

        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Calculate global ids of threads from the particular block to load from matrix B depending on i */
        //int globalPosB = offsetB + get_global_id(0) + ((i * get_local_size(0)) << TILEY_SHIFT) * get_global_size(0);
        const int col_B = get_global_id(0);
        const int row_B = i * get_local_size(0);
        const int nr_rows_B = temp;
        const int nr_cols_B = get_global_size(0); 

        /* This loop runs for number of threads in horizontal direction in the block of A */
        for(int j = 0; j < get_local_size(0) * 4; j=j+4)
        {
            /* Load 4 float4s from blockA : access patters = strided from local memory */
            float4 tempA0 = blockA[(j >> 2) + get_local_id(1) * TILEY * get_local_size(0)];
            float4 tempA1 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 1) * get_local_size(0)];
            float4 tempA2 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 2) * get_local_size(0)];
            float4 tempA3 = blockA[(j >> 2) + (get_local_id(1) * TILEY + 3) * get_local_size(0)];

            /* Load corresponding values from matrixB, access pattern = linear from global memory */
            float4 tempB0;
            float4 tempB1;
            float4 tempB2;
            float4 tempB3;

            if(!BInColMajorOrder) {
              int globalPosB = get_index(offsetB, row_B, col_B, nr_cols_B);

              tempB0 = matrixB[globalPosB  + j *  nr_cols_B]; //Should be localId.x * (TILEX / 4)
              tempB1 = matrixB[globalPosB  + (j + 1) * nr_cols_B];
              tempB2 = matrixB[globalPosB  + (j + 2) * nr_cols_B];
              tempB3 = matrixB[globalPosB  + (j + 3) * nr_cols_B];
            } else {
              // ??¿¿ No está pensado bien que OK
              int globalPosB = get_index(offsetB, col_B, row_B + (j >> 2), nr_rows_B);

              // load block in private memory
              float4 v1 = matrixB[globalPosB];
              float4 v2 = matrixB[globalPosB + 1 * nr_rows_B];
              float4 v3 = matrixB[globalPosB + 2 * nr_rows_B];
              float4 v4 = matrixB[globalPosB + 3 * nr_rows_B];

              // now we transpose it
              tempB0 = (float4) (v1.x, v2.x, v3.x, v4.x);
              tempB1 = (float4) (v1.y, v2.y, v3.y, v4.y);
              tempB2 = (float4) (v1.z, v2.z, v3.z, v4.z);
              tempB3 = (float4) (v1.w, v2.w, v3.w, v4.w);

            }
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
        sum0 = sigmoid(sum0);
        sum1 = sigmoid(sum1);
        sum2 = sigmoid(sum2);
        sum3 = sigmoid(sum3);
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
    if(sumToMatrixC) {
        const float4 a = matrixC[globalPos] * multPrevVal;
        const float4 b = matrixC[globalPos +  get_global_size(0)] * multPrevVal;
        const float4 c = matrixC[globalPos +  2 * get_global_size(0)] * multPrevVal;
        const float4 d = matrixC[globalPos +  3 * get_global_size(0)] * multPrevVal;  

        matrixC[globalPos] = a + multSum*sum0;
        matrixC[globalPos +  get_global_size(0)] = b + multSum*sum1;
        matrixC[globalPos +  2 * get_global_size(0)] = c + multSum*sum2;
        matrixC[globalPos +  3 * get_global_size(0)] = d + multSum*sum3;    
    } else {
        matrixC[globalPos] = sum0;
        matrixC[globalPos +  get_global_size(0)] = sum1;
        matrixC[globalPos +  2 * get_global_size(0)] = sum2;
        matrixC[globalPos +  3 * get_global_size(0)] = sum3;    
    }
}

/* Substracts element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSubstractKernel(__global float4 *t,
                                         __global float4 *y,
                                         __global float4* delta,
                                         int offset_t,
                                         int offset_y,
                                         int offset_delta)
{
    int i = get_global_id(0);
    
    float4 a = t[offset_t + i];
    float4 b = y[offset_y + i];
    
    delta[offset_delta + i] =  a - b;
}

/* Adds element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSumKernel(__global float4* t,
                                   __global float4* y,
                                   __global float4* delta,
                                   int offset_t,
                                   int offset_y,
                                   int offset_delta)
{
    int i = get_global_id(0);
    
    float4 a = t[offset_t + i];
    float4 b = y[offset_y + i];
    
    delta[offset_delta + i] =  a + b;
}


__kernel void elementWiseMultiplicationBySigmoidDerivativeKernel(
                                         __global float4 *del,
                                         __global float4 *act,
                                         int offset_del,
                                         int offset_act)
{
    int i = get_global_id(0);

    float4 a = sigmoid_derivative(act[offset_act + i]);
    
    del[offset_del + i] *= a;
}


__kernel void crossEntropyKernelLocal(__global float4* t, 
                                      __global float4* y, 
                                      __global float4* output, 
                                      __local float4* sdata,
                                      int offset_y)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    
    float4 y1 = y[offset_y + stride];
    float4 t1 = t[stride];
    float4 i1 = cross_entropy(t1, y1);
    
    float4 y2 = y[offset_y + stride + 1];
    float4 t2 = t[stride + 1];
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
    if(tid == 0) output[bid] = sdata[0];	
}

// Al finalizar la función se obtiene un vector de output de tamaño igual al número de grupos
// que hay que sumar, obteniendo el resultado final


/* Matrix transpose with OpenCL
* Device code.
*/

//// This kernel is optimized to ensure all global reads and writes are coalesced,
//// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
//// than the naive kernel.  Note that the shared memory array is sized to 
//// (TRANSPOSE_BLOCK_DIM+1)*TRANSPOSE_BLOCK_DIM.  This pads each row of the 2D block in shared memory 
//// so that bank conflicts do not occur when threads address the array column-wise.
//__kernel void transpose(__global float *odata, 
//                        __global float *idata, 
//                        int width, 
//                        int height, 
//                        __local float* block,
//                        int offset_o,
//                        int offset_i)
//{
//	// read the matrix tile into shared memory
//	unsigned int xIndex = get_global_id(0);
//	unsigned int yIndex = get_global_id(1);
//
//	if((xIndex < width) && (yIndex < height))
//	{
//		unsigned int index_in = offset_i + yIndex * width + xIndex;
//		block[get_local_id(1)*(TRANSPOSE_BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// write the transposed matrix tile to global memory
//	xIndex = get_group_id(1) * TRANSPOSE_BLOCK_DIM + get_local_id(0);
//	yIndex = get_group_id(0) * TRANSPOSE_BLOCK_DIM + get_local_id(1);
//	if((xIndex < height) && (yIndex < width))
//        {
//		unsigned int index_out = offset_o + yIndex * height + xIndex;
//		odata[index_out] = block[get_local_id(0)*(TRANSPOSE_BLOCK_DIM+1)+get_local_id(1)];
//	}
//}

/* 
   Returns a list of random indexes with possibility of repetition between 
   0 and max_val. The first time is called the vectors seed_x, seed_y,
   seed_z, seed_w have to be filled with NON REPEATED random elements
   (all seeds different). Global range is 1D.
   This algorithm has a maximal period of 2^128 − 1[4] 
   and passes the diehard tests. 
   However, it fails the MatrixRank and LinearComp tests of the BigCrush 
   test suite from the TestU01 framework.
 */
__kernel void random_xorshift128(uint4 * index,
                                 uint max_val,
                                 uint4 * seed_x,
                                 uint4 * seed_y,
                                 uint4 * seed_z,
                                 uint4 * seed_w) 
{
    uint i = get_global_id(0);

    uint4 t = seed_x[i] ^ (seed_x[i] << 11);
    uint4 w = seed_w[i];
    uint4 val = w ^ (w >> 19) ^ t ^ (t >> 8);

    seed_x[i] = seed_y[i]; 
    seed_y[i] = seed_z[i]; 
    seed_z[i] = seed_w[i];    
    seed_w[i] = val;

    index[i] = val % max_val;
}
