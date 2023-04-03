#ifndef h_fce_H
#define h_fce_H


#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "kernel.cu"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define num_of_agents 512
#define pow_of_agents num_of_agents*num_of_agents
#define num_of_dims 10
#define num_of_indices num_of_agents*num_of_dims
#define input_func 1
#define lo -5
#define hi 5
#define num_of_best_indices 1
#define max_iter 1000
#define num_of_agents_half num_of_agents/2
#define pow_of_agents_half pow_of_agents/2
#define num_of_runs 9

//using namespace thrust;

//class pop
//{
//	float* agent_pos = NULL;
//	float* agent_val = NULL;
//};



__global__ void get_constr(const int min, const int max, int* a, int* b);

__global__ void init_pop_pos(float* agent_pos, const int* a, const int* b, unsigned long seed);

__global__ void cost_func(const float* agent_pos, float* agent_val);
//__global__ void sphere(const float* agent_pos, float* agent_val);
//__global__ void styblinski–tang(const float* agent_pos, float* agent_val);

__global__ void DE(const float w, const float p, const int* a, const int* b,
					 const unsigned int* Ri, const unsigned int* X, const float* Rj,
					 const unsigned int* best_sol, const float* agent_pos, const float* agent_val, float* y);

__global__ void pso_f(const float w, const float c1, const float c2, const int* a, const int* b, const float* r_i,
	const unsigned int* best_sol, float* agent_pos, const float* agent_best_pos, const float* agent_val);

__global__ void ffa(const float alfa, const float beta, const float gamma, const int* a, const int* b, const float* r,
	 const float* agent_pos, float* agent_new_pos, const float* agent_val);

__global__ void GWO(const unsigned int* best_ind, const float* r_a, const int* a, const int* b,
	const float A, const float* agent_pos, float* agent_new_pos);

__global__ void iGWO_nh(unsigned int* r_w, const float* r, const int* a, const int* b,
	const float* distance, float* agent_pos, const float* agent_new_pos, float* nh_pos, unsigned int* ind_to_choose);

__global__ void calc_distances(const float* agent_pos, const float* agent_new_pos, float* distance);

__global__ void iGWO_compare_two_pop(float* pos, float* val, const float* GWO_pos, const float* GWO_val,
	const float* nh_pos, const float* nh_val);

__global__ void compare_two_pop(float* f_pos, float* f_val, const float* s_pos, const float* s_val);

__global__ void compare_ff_pos(float* old_pos, float* old_val, const float* new_pos, const float* new_val);

//__global__ void best_sol(const int num_of_agent, const float* agent_val, unsigned int* indice, float* best_val);
#endif /* h_fce_H */

