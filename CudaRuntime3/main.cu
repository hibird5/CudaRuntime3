#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "h_fce.cuh"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cub;






__global__ void searchForBestKernel(volatile float* objectiveValues, size_t* indices)
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float oVs[num_of_agents];
	__shared__ unsigned int ind[num_of_agents];

	oVs[id] = objectiveValues[id];
	oVs[id + num_of_agents_half] = objectiveValues[id + num_of_agents_half];
	ind[id] = id;
	ind[id + num_of_agents_half] = id + num_of_agents_half;
	__syncthreads();
	unsigned int step = num_of_agents_half;

#pragma unroll
	for (int i = 0; i < num_of_runs; ++i)
	{
		ind[id] = ((oVs[ind[id + step]] < oVs[ind[id]]) ? ind[id + step] : ind[id]);
		step >>= 1;
		__syncthreads();
	}
	indices[id] = ind[id];
	__syncthreads();
}



 

int main()
{
	//device init
	float* agent_pos = NULL;
	float* agent_val = NULL;
	size_t* indice = NULL;
	int* a = NULL;
	int* b = NULL;
	curandGenerator_t pseudo_rand;
	float* best_sol_a = NULL;
	float* y_DE;
	float* y_DE_val;
	float* best_DE;
	size_t* best_de;

	float* agent_best_pso;
	float* agent_best_pso_v;
	size_t* best_pso;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));

	cudaMalloc(&y_DE, num_of_indices * sizeof(float));
	cudaMalloc(&y_DE_val, num_of_agents * sizeof(float));
	cudaMalloc(&best_DE, max_iter * sizeof(float));		//vals
	cudaMalloc(&best_de, sizeof(size_t));				//indice

	cudaMalloc(&agent_best_pso, num_of_indices * sizeof(float));
	cudaMalloc(&agent_best_pso_v, num_of_agents * sizeof(float));
	cudaMalloc(&best_pso, max_iter * sizeof(float));		//vals

	cudaMalloc(&indice, num_of_agents * sizeof(size_t));
	cudaMalloc(&best_sol_a, num_of_best_indices * sizeof(float));
	cudaMalloc(&a, num_of_dims * sizeof(int));
	cudaMalloc(&b, num_of_dims * sizeof(int));

	//host init
	float* pop_back = NULL;
	float* pop_vals = NULL;
	float* best = NULL;
	size_t* ind;

	pop_back = (float*)malloc(num_of_indices * sizeof(float));
	pop_vals = (float*)malloc(num_of_agents * sizeof(float));
	best = (float*)malloc(max_iter * sizeof(float));
	ind = (size_t*)malloc(num_of_agents * sizeof(size_t));

	// prog
	get_constr << <num_of_dims, 1 >> > (-100, 100, a, b);
	init_pop_pos << <num_of_agents, num_of_dims >> > (agent_pos, a, b, time(NULL));
	cost_func << <num_of_agents, 1 >> > (agent_pos, agent_val);

	cudaMemcpy(pop_vals, agent_val, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);

	searchForBestKernel << <best_bl_th, best_bl_th >> > (agent_val, indice);
	cudaMemcpy(ind, indice, num_of_agents * sizeof(size_t), ::cudaMemcpyDeviceToHost);


	cudaError_t err; cudaError_t err1; cudaError_t err2;
	err = cudaMemcpy(best_de, &indice[0], sizeof(size_t), ::cudaMemcpyDeviceToDevice);

	//DE start
	
	
	//for (int i = 0; i < max_iter; ++i)
	//{
	//	DE << <num_of_agents, num_of_dims >> > (0.4, 0.7, a, b, time(NULL), indice, agent_pos, agent_val, y_DE);
	//	cost_func << <num_of_agents, 1 >> > (y_DE, y_DE_val);
	//	compare_two_pop << <num_of_agents, 1 >> > (agent_pos, agent_val, y_DE, y_DE_val);
	//	searchForBestKernel << <best_bl_th, best_bl_th>> > (agent_val, indice);

	//	cudaMemcpy(ind, indice, num_of_agents * sizeof(size_t), ::cudaMemcpyDeviceToHost);
	//	err = cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
	//	err1 = cudaMemcpy(best_de, &indice[0], sizeof(size_t), ::cudaMemcpyDeviceToDevice);
	//}

	//DE end


	//PSO start

	agent_best_pso = agent_pos;
	agent_best_pso_v = agent_val;
	for (int i = 0; i < max_iter; ++i)
	{
		pso_f << <num_of_agents, num_of_dims >> > (0.1, 0.25, 2, a, b, time(NULL), indice, agent_pos, agent_best_pso, agent_val);
		cost_func << <num_of_agents, 1 >> > (agent_pos, agent_val);
		compare_two_pop << <num_of_agents, 1 >> > (agent_best_pso, agent_best_pso_v, agent_pos, agent_val);
		searchForBestKernel << <best_bl_th, best_bl_th>> > (agent_best_pso_v, indice);

		cudaMemcpy(ind, indice, num_of_agents * sizeof(size_t), ::cudaMemcpyDeviceToHost);
		err = cudaMemcpy(&best[i], &agent_best_pso_v[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

	}

	//PSO end


	//err = cudaMemcpy(best, best_DE, max_iter * sizeof(float), ::cudaMemcpyDeviceToHost);

	err = cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	for (int i = 0; i < num_of_agents; ++i)
	{
		cout << i << " ";
		for (int j = 0; j < num_of_dims; ++j)
		{
			cout << pop_back[i * num_of_dims + j] << ", ";
		}
		cout << '\n' << endl;
	}
	cout << '\n' << endl;

	cout << ind[0] << ", " << endl;
	err1 = cudaMemcpy(&ind[0], best_de, sizeof(size_t), ::cudaMemcpyDeviceToHost);
	cout << ind[0] << ", " << endl;


	for (int i = 0; i < max_iter; ++i)
	{
		cout << best[i] << ", " << endl;
	}
	cout << '\n' << endl;


	free(pop_back);
	free(pop_vals);
	free(best);

	cudaFree(agent_pos);
	cudaFree(agent_val);
	cudaFree(indice);
	cudaFree(a);
	cudaFree(b);
	cudaFree(best_sol_a);
	cudaFree(y_DE);
	cudaFree(y_DE_val);
	cudaFree(best_DE);
	cudaFree(best_de);
	

	return 0;
}


//curand array
//float* r = NULL;
//cudaMalloc(&r, num_of_indices * sizeof(float));
//curandCreateGenerator(&pseudo_rand, CURAND_RNG_PSEUDO_PHILOX4_32_10);
//curandSetPseudoRandomGeneratorSeed(pseudo_rand, 1);
//curandGenerateUniform(pseudo_rand, r, num_of_indices);
//curandGenerateUniform(pseudo_rand, r, num_of_indices);
//curandGenerateUniform(pseudo_rand, r, num_of_indices);

