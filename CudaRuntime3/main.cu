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


#define num_of_agents 150
#define num_of_dims 3
#define num_of_indices 450
#define input_func 1
#define num_of_best_indices 50


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

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
	cudaMalloc(&indice, num_of_best_indices* sizeof(size_t));
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
	best = (float*)malloc(num_of_best_indices * sizeof(float));
	ind = (size_t*)malloc(num_of_best_indices* sizeof(size_t));

	// prog
	get_constr <<<num_of_dims, 1 >>> (-100, 100, a, b);
	init_pop_pos <<<num_of_agents, num_of_dims >>> (agent_pos, num_of_indices, a, b, time(NULL));
	cost_func <<<num_of_agents, 1 >>> (num_of_dims, agent_pos, input_func, agent_val);
	best_sol<<<num_of_best_indices,1>>>(num_of_agents, agent_val, indice, best_sol_a);

	
	cudaMemcpy(ind, indice, sizeof(size_t), ::cudaMemcpyDeviceToHost);
	cudaMemcpy(best, best_sol_a, sizeof(float), ::cudaMemcpyDeviceToHost);
	cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	cudaMemcpy(pop_vals, agent_val, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
	
	//curandState r1, r2, r3;
	//curand_init(1, 1, 0, &r1);
	////curand_init(seed, blockIdx.x, 0, &state);
	////curand_init(seed, blockIdx.x, 0, &state);
	//curand_uniform(&r1);
	//cout << (int)r1 << endl;

	//for (unsigned int i = 0; i < num_of_best_indices; ++i)
	//{
		//cout << rand() << ", ";
	//}
	//cout << '\n' << endl;


	for (int i = 0; i < num_of_agents; ++i)
	{
		for (int j = 0; j < num_of_dims; ++j)
		{
			cout << pop_back[i * num_of_dims + j] << ", ";
		}
		cout << '\n' << endl;
	}
	cout << '\n' << endl;

	for (int i = 0; i < num_of_agents; ++i)
	{
		cout << pop_vals[i] << ", ";
	}
	cout << '\n' << endl;

	cout << best[0] << ", " << ind[0]<< endl;

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

__global__ void DE(const float w, const float p, const int* a, const int* b,const unsigned long seed, const unsigned int best_sol,
	float* agent_pos, float* agent_val)
{
	//init	
	__shared__ float u[num_of_indices];
	float u_tmp = 0;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int i_r1, i_r2, i_r3, i_r4;
	curandState r1; // , r2, r3;
	curand_init(seed, blockIdx.x, 0, &r1);
	//curand_init(seed, blockIdx.x, 1, &r2);
	//curand_init(seed, blockIdx.x, 2, &r3);
	i_r1 = threadIdx.x + (curand(&r1) % blockDim.x) * blockDim.x;
	i_r2 = threadIdx.x + (curand(&r1) % blockDim.x) * blockDim.x;
	i_r3 = threadIdx.x + (curand(&r1) % blockDim.x) * blockDim.x;
	i_r4 = threadIdx.x + (curand(&r1) % blockDim.x) * blockDim.x;

	u_tmp = (index < num_of_indices) ?
		agent_pos[best_sol + threadIdx.x] + w * (agent_pos[i_r1] + agent_pos[i_r2] - agent_pos[i_r3] - agent_pos[i_r4]) 
		: 
		u_tmp;

	u[index] = (a[threadIdx.x] <= u_tmp) ? u_tmp : a[threadIdx.x];
	u[index] = (b[threadIdx.x] >= u_tmp) ? u_tmp : b[threadIdx.x];

	/*
	rj
	x
	if 
	...


	best pos ?
	*/



}