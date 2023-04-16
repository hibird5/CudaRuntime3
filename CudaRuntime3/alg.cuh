#ifndef alg_H
#define alg_H


#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include"h_fce.cuh"
#include <curand.h>

__host__ void Diff_ev(const float w, const float p, const float* init_pop, const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);

__host__ void PSO(const float* init_pop,const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);

__host__ void FF(const float* init_pop,const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);

__host__ void ABC(const float* init_pop,const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);

__host__ void GWO(const float* init_pop,const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);

__host__ void iGWO(const float* init_pop,const float* init_vals, const int* a, const int* b, float* best_pos, float* best_vals, float time_per_iter);


#endif /* alg_H */