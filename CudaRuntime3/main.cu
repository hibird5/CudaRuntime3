#ifndef __CUDACC__ 
#define __CUDACC__
#endif


#include "h_fce.cuh"

#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cub;



__global__ void searchForBestKernel(volatile float* objectiveValues, unsigned int* indices)
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

__global__ void searchForBestThree(volatile float* objectiveValues, unsigned int* best_three)
{
	best_three[0] = num_of_agents;
	best_three[1] = num_of_agents;

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float oVs[num_of_agents];
	__shared__ unsigned int ind[num_of_agents];

	for (int j = 0; j < 3; j++) {

		oVs[id] = objectiveValues[id];
		oVs[id + num_of_agents_half] = objectiveValues[id + num_of_agents_half];
		ind[id] = id;
		ind[id + num_of_agents_half] = id + num_of_agents_half;
		__syncthreads();
		unsigned int step = num_of_agents_half;

#pragma unroll
		for (int i = 0; i < num_of_runs; ++i)
		{	
			unsigned int ind_s = id + step;
			if (ind[id] == best_three[0] || ind[id] == best_three[1])
				ind[id] = ind[ind_s];
			else if(ind[ind_s] == best_three[0] || ind[ind_s] == best_three[1])
				ind[id] = ind[id];
			else
				ind[id] = ((oVs[ind[ind_s]] < oVs[ind[id]]) ? ind[ind_s] : ind[id]);
			step >>= 1;
			__syncthreads();
		}
		best_three[0] = (j == 0) ? ind[id] : best_three[0];
		best_three[1] = (j == 1) ? ind[id] : best_three[1];
		best_three[2] = (j == 2) ? ind[id] : best_three[2];
		__syncthreads();
	}
}

 

int main()
{
	//device init
	float* agent_pos = NULL;
	float* agent_val = NULL;
	unsigned int* indice = NULL;
	int* a = NULL;
	int* b = NULL;
	float* best_sol_a = NULL;
	float* y_DE;
	float* y_DE_val;
	float* best_DE;
	float* ff_new_poss;
	float* ff_new_vals;
	unsigned int* best_de;

	float* agent_best_pso;
	float* agent_best_pso_v;
	unsigned int* best_pso;
	float* X_gwo;
	float* nh_pos;
	float* nh_val;
	float* dist_gwo;
	unsigned int* ind_to_choose;
	float* cost_func_tmp;

	cudaMalloc(&X_gwo, num_of_indices * sizeof(float));
	cudaMalloc(&nh_pos, num_of_indices * sizeof(float));
	cudaMalloc(&nh_val, num_of_agents * sizeof(float));
	cudaMalloc(&dist_gwo, pow_of_agents * sizeof(float));
	cudaMalloc(&ind_to_choose, pow_of_agents * sizeof(unsigned int));
	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
	
	cudaMalloc(&y_DE, num_of_indices * sizeof(float));
	cudaMalloc(&y_DE_val, num_of_agents * sizeof(float));
	cudaMalloc(&best_DE, max_iter * sizeof(float));		//vals
	cudaMalloc(&best_de, sizeof(unsigned int));				//indice

	cudaMalloc(&agent_best_pso, num_of_indices * sizeof(float));
	cudaMalloc(&agent_best_pso_v, num_of_agents * sizeof(float));
	cudaMalloc(&best_pso, max_iter * sizeof(float));		//vals

	cudaMalloc(&ff_new_poss, num_of_agents*num_of_indices * sizeof(float));
	cudaMalloc(&ff_new_vals, num_of_agents * num_of_agents * sizeof(float));

	cudaMalloc(&indice, num_of_agents * sizeof(unsigned int));
	cudaMalloc(&best_sol_a, num_of_best_indices * sizeof(float));
	cudaMalloc(&a, num_of_dims * sizeof(int));
	cudaMalloc(&b, num_of_dims * sizeof(int));

	cudaMalloc(&cost_func_tmp, pow_of_agents * dims_to_log_half * sizeof(float));

	//host init
	float* pop_back = NULL;
	float* pop_vals = NULL;
	float* best = NULL;
	float* ff = NULL;
	unsigned int* ind;
	unsigned int* indi;
	ff = (float*)malloc(num_of_agents * num_of_agents * num_of_dims * sizeof(float));
	pop_back = (float*)malloc(num_of_indices * sizeof(float));
	pop_vals = (float*)malloc(num_of_agents * sizeof(float));
	best = (float*)malloc(max_iter * sizeof(float));
	ind = (unsigned int*)malloc(num_of_agents * sizeof(unsigned int));
	indi = (unsigned int*)malloc(num_of_agents * sizeof(unsigned int));
	// prog

	get_constr << <num_of_dims, 1 >> > (lo, hi, a, b);
	init_pop_pos << <num_of_agents, num_of_dims >> > (agent_pos, a, b, (unsigned long)time(NULL));
	cost_func << <num_of_agents, dims_to_log_half >> > (agent_pos, agent_val, cost_func_tmp);
	//thrust::sort(thrust::device, agent_val, agent_val);

	//
	//cudaMemcpy(pop_vals, agent_val, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
	//	for (int i = 0; i < num_of_agents; ++i)
	//	{
	//		cout << pop_vals[i] << ", " << endl;
	//	}
	 //cudaMemcpy(pop_vals, agent_val, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
	 //for (int i = 0; i < num_of_agents; ++i)
	 //{
		// cout << i << ".    " << pop_vals[i] << ", " << endl;
	 //}
	 //cout << endl;

	 //cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	 //for (int i = 0; i < num_of_agents; ++i)
	 //{
 	//	cout << i << " ";
 	//	for (int j = 0; j < num_of_dims; ++j)
 	//	{
 	//		cout << pop_back[i * num_of_dims + j] << ", ";
 	//	}
 	//	cout << '\n' << endl;
	 //}
	 //cout << '\n' << endl;
	//searchForBestKernel << <1, num_of_agents_half >> > (agent_val, indice);

	cudaError_t err; cudaError_t err1; cudaError_t err2;
	//err = cudaMemcpy(best_de, &indice[0], sizeof(unsigned int), ::cudaMemcpyDeviceToDevice);

	//thrust::sort(thrust::device, agent_val, agent_val + num_of_agents);
	err = cudaGetLastError();

	//DE start
	
	//unsigned int* r;
	//unsigned int* X;
	//float* Rj;
	//curandGenerator_t r_int;
	//unsigned int num_of_Ri = 4 * num_of_indices;
	//cudaMalloc(&r, num_of_Ri * sizeof(unsigned int));
	//cudaMalloc(&X, num_of_indices * sizeof(unsigned int));
	//cudaMalloc(&Rj, num_of_indices * sizeof(float));
	//
	//curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	//curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
	//
	//for (int i = 0; i < max_iter; ++i)
	//{
	//	curandGenerate(r_int, r, num_of_Ri);
	//	curandGenerate(r_int, X, num_of_indices);
	//	curandGenerateUniform(r_int, Rj, num_of_indices);
	//
	//	DE << <  num_of_agents, num_of_dims>> > (0.4, 0.7, a, b, r, X, Rj,
	//											indice, agent_pos, agent_val, y_DE);
	//	cudaDeviceSynchronize();
	//	cost_func << <1, num_of_agents >> > (y_DE, y_DE_val);
	//	cudaDeviceSynchronize();
	//	compare_two_pop << <num_of_agents, num_of_dims >> > (agent_pos, agent_val, y_DE, y_DE_val);
	//	cudaDeviceSynchronize();
	//	searchForBestKernel << <1, num_of_agents_half>> > (agent_val, indice);
	//
	//	cudaMemcpy(ind, indice, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
	//	err = cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
	//	err1 = cudaMemcpy(best_de, &indice[0], sizeof(unsigned int), ::cudaMemcpyDeviceToDevice);
	//}
	
	//DE end
	
	//FF start

	//dim3 agents(num_of_agents, num_of_agents, 1);
	//float* r_a;
	//curandGenerator_t r_in;
	//unsigned int num_of_uR = pow_of_agents * num_of_dims;
	//cudaMalloc(&r_a, num_of_uR * sizeof(float));
	//
	//curandCreateGenerator(&r_in, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	//curandSetPseudoRandomGeneratorSeed(r_in, time(NULL));
	//
	//
	////searchForBestFF<< <num_of_agents_half, num_of_agents>> > (ff_new_vals, indice);
	//for (int i = 0; i < max_iter; ++i)
	//{
	//	curandGenerateNormal(r_in, r_a, num_of_uR, 0.0, 0.5);
	//	ffa << <agents, num_of_dims >> > (1, 1, 1/10, a, b, r_a, agent_pos, ff_new_poss, agent_val);
	//	cudaDeviceSynchronize();
	//	cost_func << <agents, dims_to_log_half>> > (ff_new_poss, ff_new_vals, cost_func_tmp);
	//
	//
	//	cudaDeviceSynchronize();
	//	compare_ff_pos << <num_of_agents, num_of_dims >> > (agent_pos, agent_val, ff_new_poss, ff_new_vals);
	//	cudaDeviceSynchronize();
	//	
	//	//cudaMemcpy(pop_vals, agent_val, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
	//	//for (int i = 0; i < num_of_agents; ++i)
	//	//{
	//	//	cout << i << ".    " << pop_vals[i] << ", " << endl;
	//	//}
	//	//cout << endl;
	//
	//	cudaMemcpy(pop_back, ff_new_poss, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	//	for (int i = 0; i < num_of_agents; ++i)
	//	{
	//		cout << i << " ";
	//		for (int j = 0; j < num_of_dims; ++j)
	//		{
	//			cout << pop_back[i * num_of_dims + j] << ", ";
	//		}
	//		cout << '\n' << endl;
	//	}
	//	cout << '\n' << endl;
	//
	//	searchForBestKernel << <1, num_of_agents_half >> > (agent_val, indice);
	//
	//	cudaMemcpy(ind, indice, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
	//	cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
	//	err = cudaGetLastError();
	//}

	//FF end

	//PSO start
	
	//float* r;
	//unsigned int num_of_ri = 2 * num_of_indices;
	//curandGenerator_t r_int;
	//cudaMalloc(&r, num_of_ri * sizeof(float));
	//
	//curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	//curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
	//
	//agent_best_pso = agent_pos;
	//agent_best_pso_v = agent_val;
	//for (int i = 0; i < max_iter; ++i)
	//{
	//	curandGenerateUniform(r_int, r, num_of_ri);
	//	pso_f << <  num_of_agents, num_of_dims>> > (0.1, 0.25, 2, a, b, r, indice, agent_pos, agent_best_pso, agent_val);
	//	cost_func << <num_of_agents, dims_to_log_half >> > (agent_pos, agent_val);
	//	compare_two_pop << <num_of_agents, num_of_dims >> > (agent_best_pso, agent_best_pso_v, agent_pos, agent_val);
	//	searchForBestKernel << <1, num_of_agents_half >> > (agent_best_pso_v, indice);
	//	cudaMemcpy(ind, indice, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
	//	err = cudaMemcpy(&best[i], &agent_best_pso_v[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
	//}
	
	//PSO end



	//iGWO start
/*
	dim3 agents(num_of_agents, num_of_agents, 1);

	float* r_a; float* r_d; float* r;
	unsigned int* r_w; unsigned int* r_nh;
	curandGenerator_t r_in;
	unsigned int num_of_uR = pow(num_of_agents, 2);
	cudaMalloc(&r_a, 3 * num_of_agents * sizeof(float));
	cudaMalloc(&r_d, 3 * num_of_agents * sizeof(float));
	cudaMalloc(&r, num_of_indices * sizeof(float));

	cudaMalloc(&r_w, num_of_agents* sizeof(unsigned int));
	cudaMalloc(&r_nh, num_of_agents * sizeof(unsigned int));

	curandCreateGenerator(&r_in, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_in, time(NULL));

	curandGenerateUniform(r_in, r_a, 6 * num_of_agents);
	curandGenerateUniform(r_in, r, num_of_indices);
	curandGenerate(r_in, r_w, 2 * num_of_agents);

	cudaError_t eer;
	float A = 0;
	double aaa;
	auto s = std::chrono::high_resolution_clock::now();;
	long long e = 0; long long ee = 0; long long eee = 0; long long eeee = 0; long long eeeee = 0; long long eew = 0; long long eeew = 0;
	for (int i = 0; i < max_iter; ++i) 
	{
		A = 2 - 2 * i / max_iter;

		searchForBestThree << <1, num_of_agents_half >> > (agent_val, indice);

		//cudaMemcpy(indi, indice, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);

		//err = cudaMemcpy(&best[i], &agent_val[indi[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

		GWO << < num_of_agents, num_of_dims>> > (indice, r_a, a, b, A, agent_pos, X_gwo);

		calc_distances<<<agents,1>>>(agent_pos, dist_gwo);

		iGWO_nh << <agents, num_of_dims>> > (r_w, r, a, b, dist_gwo, agent_pos, X_gwo, nh_pos, ind_to_choose);

		cost_func << <num_of_agents, dims_to_log_half >> > (nh_pos, nh_val);
	
		cost_func << <num_of_agents, dims_to_log_half>> > (X_gwo, agent_val);
		
		eer = cudaGetLastError();

		//cudaDeviceSynchronize();
		compare_two_pop << <num_of_agents, num_of_dims >> > (agent_pos, agent_val, X_gwo, agent_val, nh_pos, nh_val);
		curandGenerateUniform(r_in, r_a, 6 * num_of_agents);
		curandGenerateUniform(r_in, r, num_of_indices);
		curandGenerate(r_in, r_w, 2 * num_of_agents);


		cudaMemcpy(indi, dist_gwo, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
			for (int i = 0; i <num_of_agents; ++i)
			{
				cout << indi[i] << ", " << endl;
			}
		cout << '\n' << endl;
	}*/
	/*	s = std::chrono::high_resolution_clock::now();
		eeew += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();
	e /= 1000; ee /= 1000; eee /= 1000; eeee /= 1000; eeeee /= 1000; eeew /= 1000; eew /= 1000;*/
	//igwo end

	//abc start

	float* r;
	unsigned int* rI;
	unsigned int* abbadon_dec;
	unsigned int* index_to_rns;
	curandGenerator_t r_int;
	cudaMalloc(&r, num_of_indices * sizeof(float));
	cudaMalloc(&rI, num_of_agents * sizeof(float));
	cudaMalloc(&index_to_rns, num_of_agents * sizeof(unsigned int));
	cudaMalloc(&abbadon_dec, num_of_agents * sizeof(unsigned int));
	curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
	

	unsigned int abbadon_val = max_iter / 2;

	for (int i = 0; i < max_iter; ++i)
	{
		curandGenerate(r_int, rI, num_of_agents);
		curandGenerateUniform(r_int, r, num_of_indices);

		abc_rns << <num_of_agents, num_of_dims >> > (agent_pos, agent_best_pso, a, b, r, rI);
		cost_func << <num_of_agents, dims_to_log_half >> > (agent_best_pso, agent_best_pso_v, cost_func_tmp);
		compare_two_pop << <num_of_agents, num_of_dims >> > (agent_pos, agent_val, agent_best_pso, agent_best_pso_v, abbadon_dec);
		probability_selection << <1, num_of_agents >> > (agent_val, r, index_to_rns);
		err = cudaGetLastError();
		//err2 = cudaMemcpy(ind, index_to_rns, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		//for (int i = 0; i < num_of_agents; ++i)
		//{
		//	cout << ind[i] << ", " << endl;
		//}

		curandGenerate(r_int, rI, num_of_agents);
		curandGenerateUniform(r_int, r, num_of_indices);
		abc_rns << <num_of_agents, num_of_dims >> > (agent_pos, agent_best_pso, index_to_rns, a, b, r, rI);
		err = cudaGetLastError();
		cost_func << <num_of_agents, dims_to_log_half >> > (agent_best_pso, agent_best_pso_v, cost_func_tmp);
		compare_two_pop << <num_of_agents, num_of_dims >> > (agent_pos, agent_val, agent_best_pso, agent_best_pso_v, abbadon_dec);

		curandGenerateUniform(r_int, r, num_of_indices);
		err = cudaGetLastError();
		scout_phase<<<num_of_agents, num_of_dims>>>(abbadon_dec, 2, a, b, r, agent_pos);
		err2 = cudaMemcpy(pop_back, agent_pos, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
		err = cudaGetLastError();
		cost_func << <num_of_agents, dims_to_log_half >> > (agent_pos, agent_val, cost_func_tmp);
		err = cudaGetLastError();
		searchForBestKernel << <1, num_of_agents_half>> > (agent_val, indice);
		err2 = cudaMemcpy(ind, indice, num_of_agents * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		err1 = cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

		err = cudaGetLastError();

	}



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


	//err = cudaMemcpy(pop_back, agent_pos, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToHost);
	//for (int i = 0; i < num_of_agents; ++i)
	//{
	//	cout << i << " ";
	//	for (int j = 0; j < num_of_dims; ++j)
	//	{
	//		cout << pop_back[i * num_of_dims + j] << ", ";
	//	}
	//	cout << '\n' << endl;
	//}
	//cout << '\n' << endl;
	
		//err2 = cudaMemcpy(pop_vals, agent_pos, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToHost);
		//for (int i = 0; i < num_of_agents; ++i)
		//{
		//	cout << pop_vals[i] << ", " << endl;
		//}
		
	//int* gg;
	//cudaMalloc(&gg, 4 * sizeof(int));
	//int *gge = (int*)malloc(4* sizeof(int));
	//tet << <agents, block>> > (gg);
	//err =cudaMemcpy(gge, gg, 4 * sizeof(int), ::cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 4; ++i)
	//{
	//	cout << gge[i] << endl;
	//}