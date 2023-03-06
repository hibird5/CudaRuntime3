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


#define num_of_agents 1000
#define num_of_dims 5
#define num_of_indices num_of_agents*num_of_dims
#define input_func 10
#define num_of_best_indices 5
#define max_iter 100

__global__ void DE(const float w, const float p, const int* a, const int* b, const unsigned long seed, const size_t* best_sol,
	const float* agent_pos,const float* agent_val, float* y)
{
	//init	
	float u_tmp = 0;
	float u;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int i_r1, i_r2, i_r3, i_r4, X;
	float Rj;
	curandState r1; // , r2, r3;
	curand_init(seed, blockIdx.x, 0, &r1);
	//curand_init(seed, blockIdx.x, 1, &r2);
	//curand_init(seed, blockIdx.x, 2, &r3);
	i_r1 = (curand(&r1) % blockDim.x) * blockDim.x;
	i_r2 = (curand(&r1) % blockDim.x) * blockDim.x;
	i_r3 = (curand(&r1) % blockDim.x) * blockDim.x;
	i_r4 = (curand(&r1) % blockDim.x) * blockDim.x;

	u_tmp = (index < num_of_indices) ?
		agent_pos[best_sol[0]+ threadIdx.x] + w * (agent_pos[i_r1] + agent_pos[i_r2] - agent_pos[i_r3] - agent_pos[i_r4])
		//agent_pos[i_r1] + w * (agent_pos[i_r2] - agent_pos[i_r3])
		:
		u_tmp;

	//search dom test
	u = (a[threadIdx.x] <= u_tmp) ? u_tmp : a[threadIdx.x];
	u = (b[threadIdx.x] >= u_tmp) ? u_tmp : b[threadIdx.x];


	__syncthreads();
	Rj = curand_uniform(&r1);
	X = curand(&r1) % num_of_dims;

	y[index] = (Rj <= p ) ? u : agent_pos[index];
	y[index] = (X == threadIdx.x) ? u : agent_pos[index];
	__syncthreads();
}

__global__ void compare_two_pop(float* f_pos, float* f_val, const float* s_pos, const float* s_val)
{
	//f_val[blockIdx.x] = (s_val[blockIdx.x] < f_val[blockIdx.x]) ? s_val[blockIdx.x] : f_val[blockIdx.x];
	int ind;

	if (s_val[blockIdx.x] < f_val[blockIdx.x])
	{
		f_val[blockIdx.x] = s_val[blockIdx.x];

#pragma unroll
		for (int i = 0; i < num_of_dims; ++i)
			ind = i + blockIdx.x * blockDim.x;
		f_pos[ind] = s_pos[ind];

	}
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

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));

	cudaMalloc(&y_DE, num_of_indices * sizeof(float));
	cudaMalloc(&y_DE_val, num_of_agents * sizeof(float));
	cudaMalloc(&best_DE, max_iter * sizeof(float));		//vals
	cudaMalloc(&best_de, sizeof(size_t));				//indice

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
	best = (float*)malloc(max_iter * sizeof(float));
	ind = (size_t*)malloc( sizeof(size_t));

	// prog
	get_constr <<<num_of_dims, 1 >>> (-10, 10, a, b);
	init_pop_pos <<<num_of_agents, num_of_dims >>> (agent_pos, num_of_indices, a, b, time(NULL));
	cost_func <<<num_of_agents, 1 >>> (num_of_dims, agent_pos, input_func, agent_val);
	best_sol<<<num_of_best_indices,1>>>(num_of_agents, agent_val, indice, best_sol_a);

	cudaError_t err; cudaError_t err1; cudaError_t err2;
	err =cudaMemcpy(best_de, &indice[0], sizeof(size_t), ::cudaMemcpyDeviceToDevice);

	//DE start
	for (int i = 0; i < max_iter; ++i)
	{
		DE <<<num_of_agents, num_of_dims >>> (0.3, 0.8, a, b, time(NULL), best_de, agent_pos, agent_val, y_DE);
		//err = cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
		//for (int i = 0; i < num_of_agents; ++i)
		//{
		//	for (int j = 0; j < num_of_dims; ++j)
		//	{
		//		cout << pop_back[i * num_of_dims + j] << ", ";
		//	}
		//	cout << '\n' << endl;
		//}
		//cout << '\n' << endl;

		cost_func <<<num_of_agents, 1 >>> (num_of_dims, y_DE, input_func, y_DE_val);
		//err = cudaMemcpy(pop_back, y_DE, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
		//for (int i = 0; i < num_of_agents; ++i)
		//{
		//	for (int j = 0; j < num_of_dims; ++j)
		//	{
		//		cout << pop_back[i * num_of_dims + j] << ", ";
		//	}
		//	cout << '\n' << endl;
		//}
		//cout << '\n' << endl;
		compare_two_pop <<<num_of_agents, 1 >>> (agent_pos, agent_val, y_DE, y_DE_val);
		
		best_sol <<<num_of_best_indices, 1 >>> (num_of_agents, agent_val, indice, best_sol_a);
		//err2 = cudaMemcpy(&best_DE[i], &best_sol_a[0], sizeof(float), ::cudaMemcpyDeviceToDevice);
		//cudaMemcpy(&best[0], &best_sol_a[0], sizeof(float), ::cudaMemcpyDeviceToHost);
		//cout << best[0] << endl;
	

		best_sol <<<num_of_best_indices, 1 >>> (num_of_agents, y_DE_val, indice, best_sol_a);
	/*	cudaMemcpy(&best[0], &best_sol_a[0], sizeof(float), ::cudaMemcpyDeviceToHost);
		cout << best[0] << endl;
		cout << endl;*/


		err1 = cudaMemcpy(best_de, &indice[0], sizeof(size_t), ::cudaMemcpyDeviceToDevice);
		}
	//DE end
	
	cudaMemcpy(best, best_DE, max_iter * sizeof(float), ::cudaMemcpyDeviceToHost);
	
	err = cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	//for (int i = 0; i < num_of_agents; ++i)
	//{
	//	for (int j = 0; j < num_of_dims; ++j)
	//	{
	//		cout << pop_back[i * num_of_dims + j] << ", ";
	//	}
	//	cout << '\n' << endl;
	//}
	//cout << '\n' << endl;



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

