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

	cudaMalloc(&DEV_init_pop, num_of_indices * sizeof(float));
	cudaMalloc(&DEV_init_vals, num_of_agents * sizeof(float));

	cudaMalloc(&DEV_a, num_of_dims * sizeof(int));
	cudaMalloc(&DEV_b, num_of_dims * sizeof(int));

	cudaMalloc(&DEV_best_pos, num_of_dims * sizeof(float));
	cudaMalloc(&DEV_best_vals, max_iter * sizeof(float));
	// Host vars
	float time_iter = .0;
	float* H_best_vals = (float*)malloc(max_iter * sizeof(float));



	get_constr <<<num_of_dims, 1 >>> (lo, hi, DEV_a, DEV_b);
	init_pop_pos <<<num_of_agents, num_of_dims >>> (DEV_init_pop, DEV_a, DEV_b, (unsigned long)time(NULL));
	cost_func <<<num_of_agents, dims_to_log_half >>> (DEV_init_pop, DEV_init_vals);

	Diff_ev(0.4, 0.7, DEV_init_pop, DEV_init_vals, DEV_a, DEV_b, DEV_best_pos, DEV_best_vals, time_iter);





	cudaMemcpy(H_best_vals, DEV_best_vals, max_iter * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);

	for (int i = 0; i < max_iter; ++i)
	{
		cout << i << ".   " << H_best_vals[i] << ", " << endl;
		//for (int j = 0; j < num_of_dims; ++j)
		//{
		//	cout << pop_back[i * num_of_dims + j] << ", ";
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