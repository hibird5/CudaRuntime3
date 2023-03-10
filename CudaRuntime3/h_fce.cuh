#ifndef h_fce_H
#define h_fce_H


#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <cuda.h>
#include <curand.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

//class pop
//{
//	float* agent_pos = NULL;
//	float* agent_val = NULL;
//};

__global__ void get_constr(const int min, const int max, int* a, int* b);

__global__ void init_pop_pos(float* agent_pos, const int num_of_indices, const int* a, const int* b, unsigned long seed);

__global__ void cost_func(const int num_of_dims, const float* agent_pos, const int input_func, float* agent_val);

__global__ void best_sol(const int num_of_agent, const float* agent_val, size_t* indice, float* best_val);

//__global__ void DE(const float w, const float p, const int input_func, const int num_of_dims, const int a, const int b,
//			float* agent_pos, float* agent_val);





//__global__ void init_pop_vals(population* pop)
//{
//	cost_func(pop->agent_pos, pop->agent_val[blockIdx.x], pop->sizes.dimensions, pop->input_func);
//}
#endif /* h_fce_H */

