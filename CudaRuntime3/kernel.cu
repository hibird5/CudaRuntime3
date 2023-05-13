#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include "h_fce.cuh"
#include "alg.cuh"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cub;



int main()
{
	// Device vars
	float* DEV_init_pop;
	float* DEV_init_vals;

	int* DEV_a;
	int* DEV_b;

	float* DEV_best_pos;
	float* DEV_best_vals;

	cudaMalloc(&DEV_init_pop, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&DEV_init_vals, NUM_OF_AGENTS * sizeof(float));

	cudaMalloc(&DEV_a, NUM_OF_DIMS * sizeof(int));
	cudaMalloc(&DEV_b, NUM_OF_DIMS * sizeof(int));

	cudaMalloc(&DEV_best_pos, NUM_OF_DIMS * sizeof(float));
	cudaMalloc(&DEV_best_vals, MAX_ITER * sizeof(float));
	// Host vars
	float time_iter = .0;
	float* H_best_vals = (float*)malloc(MAX_ITER * sizeof(float));

	cudaError_t err = (cudaError_t)0;

	
	get_constr <<<NUM_OF_DIMS, 1 >>> (lo, hi, DEV_a, DEV_b);
	init_pop_pos <<<NUM_OF_AGENTS, NUM_OF_DIMS >>> (DEV_init_pop, DEV_a, DEV_b, (unsigned long)time(NULL));
	cost_func <<<NUM_OF_AGENTS, DIMS_TO_LOG_HALF >>> (DEV_init_pop, DEV_init_vals);

	err = cudaGetLastError();
	error_h(err);

//	Diff_ev(0.4, 0.7, DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);
//	PSO(0.1, 0.25, 2, DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);
//	FF(1, 1, 0.1, DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);
//	GWO(DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);
	iGWO(DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);



	cudaMemcpy(H_best_vals, DEV_best_vals, MAX_ITER * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);

	for (int i = 0; i < MAX_ITER; ++i)
	{
		cout << i << ".   " << H_best_vals[i] << ", " << endl;
		//for (int j = 0; j < NUM_OF_DIMS; ++j)
		//{
		//	cout << pop_back[i * NUM_OF_DIMS + j] << ", ";
		//}
		//	cout << endl;
	}
	cout << '\n' << endl;
	
	cudaFree(DEV_init_pop);
	cudaFree(DEV_init_vals);
	cudaFree(DEV_a);
	cudaFree(DEV_b);
	cudaFree(DEV_best_pos);
	cudaFree(DEV_best_vals);

	return 0;

}