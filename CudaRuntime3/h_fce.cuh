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

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>


#define NUM_OF_AGENTS 256
#define NUM_OF_RUNS 8
#define POW_OF_AGENTS NUM_OF_AGENTS*NUM_OF_AGENTS
#define NUM_OF_DIMS 2
#define NUM_OF_RUNS_ADD 1
#define DIMS_TO_LOG 2
#define DIMS_TO_LOG_HALF DIMS_TO_LOG/2

#define NUM_OF_INDICES NUM_OF_AGENTS*NUM_OF_DIMS
#define input_func SPHEERE
#define lo -500
#define hi 500
#define MAX_ITER 1000
#define NUM_OF_AGENTS_HALF NUM_OF_AGENTS/2
#define POW_OF_AGENTS_HALF POW_OF_AGENTS/2
#define ROSENBROCK 3
#define SPHEERE 2
#define ST 1


__global__ void get_constr(const int min, const int max, int* a, int* b);

__global__ void init_pop_pos(float* agent_pos, const int* a, const int* b, unsigned long seed);

__global__ void cost_func(const float* agent_pos, float* agent_val);

__global__ void cost_func(const float* agent_pos, float* agent_val, float* tmp);

//__device__ float sphere(const float& agent_pos);
//__device__ float styblinski–tang(const float& agent_pos);

__global__ void searchForBestKernel(volatile float* objectiveValues, unsigned int* indices);

__global__ void searchForBestThree(volatile float* objectiveValues, unsigned int* best_three);

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

__global__ void abc_rns(const float* agent_pos, float* agent_new_pos, const int* a, const int* b, const float* r, const unsigned int* rI);

__global__ void abc_rns(const float* agent_pos, float* agent_new_pos, const unsigned int* indices_to_compute,
	const int* a, const int* b, const float* r, const unsigned int* rI);

//__global__ void calc_distances(const float* agent_pos, const float* agent_new_pos, float* distance);

__global__ void calc_distances(const float* agent_pos, float* tmp_distance, float* distance);

__global__ void compare_two_pop(float* pos, float* val, const float* GWO_pos, const float* GWO_val,
	const float* nh_pos, const float* nh_val);

__global__ void compare_two_pop(float* f_pos, float* f_val, const float* s_pos, const float* s_val);

__global__ void compare_two_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val, unsigned int* abbadon_dec);

__global__ void compare_ff_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val);

__global__ void probability_selection(const float* val, const float* r, unsigned int* index_to_rns);

__global__ void scout_phase(unsigned int* abbadon_dec, const unsigned int abbadon_val, const int* a, const int* b,
	const float* r, float* agent_pos, unsigned int best_index);

void error_h(cudaError_t e);
//__global__ void best_sol(const int num_of_agent, const float* agent_val, unsigned int* indice, float* best_val);
#endif /* h_fce_H */

