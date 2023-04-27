//#ifndef __CUDACC__ 
//#define __CUDACC__
//#endif
//
//
//#include "h_fce.cuh"
//
//#include <stdio.h>
//#include <fstream>
//#include <iostream>
//#include <time.h>
//
//using namespace std;
//using namespace cub;
//
//
//void save_pop(float* pop_back, unsigned int  num) {
//	ofstream file;
//	string path = "../log" + to_string(num) + ".csv";
//	file.open(path, ios::out);
//	if (!file.is_open()) {
//		cout << "Failed to open a file!" << endl;
//		return;
//	}
//	for (auto i = 0; i < NUM_OF_AGENTS; i++)
//	{
//		for (auto j = 0; j < NUM_OF_DIMS; j++)
//		{
//			file <<
//				pop_back[i * NUM_OF_DIMS + j] << ",";
//		}
//		file << std::endl;
//	}
//	if (file.is_open())
//		file.close();
//}
//
//int main()
//{
//
//
//	//device init
//	float* agent_pos = NULL;
//	float* agent_val = NULL;
//	unsigned int* indice = NULL;
//	int* a = NULL;
//	int* b = NULL;
//	float* best_sol_a = NULL;
//	float* y_DE;
//	float* y_DE_val;
//	float* best_DE;
//	float* ff_new_poss;
//	float* ff_new_vals;
//	unsigned int* best_de;
//
//	float* agent_best_pso;
//	float* agent_best_pso_v;
//	unsigned int* best_pso;
//	float* X_gwo;
//	float* nh_pos;
//	float* nh_val;
//	float* dist_gwo;
//	unsigned int* ind_to_choose;
//	float* cost_func_tmp;
//
//	cudaMalloc(&X_gwo, NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&nh_pos, NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&nh_val, NUM_OF_AGENTS * sizeof(float));
//	cudaMalloc(&dist_gwo, POW_OF_AGENTS * sizeof(float));
//	cudaMalloc(&ind_to_choose, POW_OF_AGENTS * sizeof(unsigned int));
//	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));
//	
//	cudaMalloc(&y_DE, NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&y_DE_val, NUM_OF_AGENTS * sizeof(float));
//	cudaMalloc(&best_DE, MAX_ITER * sizeof(float));		//vals
//	cudaMalloc(&best_de, sizeof(unsigned int));				//indice
//
//	cudaMalloc(&agent_best_pso, NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&agent_best_pso_v, NUM_OF_AGENTS * sizeof(float));
//	//cudaMalloc(&best_pso, MAX_ITER * sizeof(float));		//vals
//
//	cudaMalloc(&ff_new_poss, NUM_OF_AGENTS*NUM_OF_INDICES * sizeof(float));
//	cudaMalloc(&ff_new_vals, NUM_OF_AGENTS * NUM_OF_AGENTS * sizeof(float));
//
//	cudaMalloc(&indice, NUM_OF_AGENTS * sizeof(unsigned int));
//	//cudaMalloc(&best_sol_a, num_of_best_indices * sizeof(float));
//	cudaMalloc(&a, NUM_OF_DIMS * sizeof(int));
//	cudaMalloc(&b, NUM_OF_DIMS * sizeof(int));
//
//	cudaMalloc(&cost_func_tmp, NUM_OF_AGENTS * DIMS_TO_LOG_HALF * sizeof(float));
//
//	float* tmp_distance;
//	cudaMalloc(&tmp_distance, POW_OF_AGENTS * DIMS_TO_LOG_HALF * sizeof(float));
//
//	//host init
//	float* pop_back = NULL;
//	float* pop_vals = NULL;
//	float* best = NULL;
//	float* ff = NULL;
//	unsigned int* ind;
//	unsigned int* indi;
//	ff = (float*)malloc(NUM_OF_AGENTS * NUM_OF_AGENTS * NUM_OF_DIMS * sizeof(float));
//	pop_back = (float*)malloc(NUM_OF_INDICES * sizeof(float));
//	pop_vals = (float*)malloc(NUM_OF_AGENTS * sizeof(float));
//	best = (float*)malloc(MAX_ITER * sizeof(float));
//	ind = (unsigned int*)malloc(NUM_OF_AGENTS * sizeof(unsigned int));
//	indi = (unsigned int*)malloc(NUM_OF_AGENTS * sizeof(unsigned int));
//	// prog
//
//	get_constr << <NUM_OF_DIMS, 1 >> > (lo, hi, a, b);
//	init_pop_pos << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, a, b, (unsigned long)time(NULL));
//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (agent_pos, agent_val, cost_func_tmp);
//
//	dim3 agents(NUM_OF_AGENTS, NUM_OF_AGENTS, 1);
//	calc_distances<<<agents, DIMS_TO_LOG_HALF>>>(agent_pos, tmp_distance, dist_gwo);
//
//	cudaMemcpy(pop_back, dist_gwo, 5 * sizeof(float), ::cudaMemcpyDeviceToHost);
//	cudaMemcpy(pop_vals, tmp_distance, 5 * sizeof(float), ::cudaMemcpyDeviceToHost);
//
//	for (int j = 0; j < 5; ++j)
//				{
//					cout<< pop_back[j] << ", " << pop_vals[j] << endl;
//				}
//
//	//thrust::sort(thrust::device, agent_val, agent_val);
//	cudaError_t err; cudaError_t err1; cudaError_t err2;
//
//
//
//	//err = cudaMemcpy(best_de, &indice[0], sizeof(unsigned int), ::cudaMemcpyDeviceToDevice);
//
//	//thrust::sort(thrust::device, agent_val, agent_val + NUM_OF_AGENTS);
//	err = cudaGetLastError();
//
//	//DE start
//	//
//	//unsigned int* r;
//	//unsigned int* X;
//	//float* Rj;
//	//curandGenerator_t r_int;
//	//unsigned int num_of_Ri = 4 * NUM_OF_INDICES;
//	//cudaMalloc(&r, num_of_Ri * sizeof(unsigned int));
//	//cudaMalloc(&X, NUM_OF_INDICES * sizeof(unsigned int));
//	//cudaMalloc(&Rj, NUM_OF_INDICES * sizeof(float));
//	//
//	//curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
//	//curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
//	//
//	//for (int i = 0; i < MAX_ITER; ++i)
//	//{
//	//	curandGenerate(r_int, r, num_of_Ri);
//	//	curandGenerate(r_int, X, NUM_OF_INDICES);
//	//	curandGenerateUniform(r_int, Rj, NUM_OF_INDICES);
//	//
//	//	DE << <  NUM_OF_AGENTS, NUM_OF_DIMS>> > (0.4, 0.7, a, b, r, X, Rj,
//	//											indice, agent_pos, agent_val, y_DE);
//	//	cudaDeviceSynchronize();
//	//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF>> > (y_DE, y_DE_val,cost_func_tmp);
//
//	//	err = cudaMemcpy(pop_back, agent_pos, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToHost);
//	//	err = cudaMemcpy(pop_vals, agent_val, NUM_OF_AGENTS* sizeof(float), ::cudaMemcpyDeviceToHost);
//	//	for (int i = 0; i < NUM_OF_AGENTS; ++i)
//	//	{
//	//		cout << i << "   " << pop_vals[i] << "----- " ;
//	//		for (int j = 0; j < NUM_OF_DIMS; ++j)
//	//		{
//	//			cout<< pop_back[i * NUM_OF_DIMS + j] << ", ";
//	//		}
//	//		cout << '\n' << endl;
//	//	}
//	//	cout << '\n' << endl;
//
//	//	compare_two_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, y_DE, y_DE_val);
//	//	cudaDeviceSynchronize();
//	//	searchForBestKernel << <1, NUM_OF_AGENTS_HALF>> > (agent_val, indice);
//	//	cudaMemcpy(ind, indice, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//	//	err = cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
//
//	//	error_h(cudaGetLastError());
//	//}
//	//
//	//DE end
//	
//	//FF start
//
//	
//	float* r_a;
//	curandGenerator_t r_in;
//	unsigned int num_of_uR = POW_OF_AGENTS * NUM_OF_DIMS;
//	cudaMalloc(&r_a, num_of_uR * sizeof(float));
//	
//	curandCreateGenerator(&r_in, CURAND_RNG_PSEUDO_PHILOX4_32_10);
//	curandSetPseudoRandomGeneratorSeed(r_in, time(NULL));
//	
//	
//	//searchForBestFF<< <NUM_OF_AGENTS_HALF, NUM_OF_AGENTS>> > (ff_new_vals, indice);
//	for (int i = 0; i < MAX_ITER; ++i)
//	{
//		curandGenerateNormal(r_in, r_a, num_of_uR, 0.0, 0.5);
//		ffa << <agents, NUM_OF_DIMS >> > (1, 1, 1/10, a, b, r_a, agent_pos, ff_new_poss, agent_val);
//		//cudaDeviceSynchronize();
//		cost_func << <agents, DIMS_TO_LOG_HALF>> > (ff_new_poss, ff_new_vals, cost_func_tmp);
//	
//	
//		//cudaDeviceSynchronize();
//		compare_ff_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, ff_new_poss, ff_new_vals);
//		cudaDeviceSynchronize();
//		
//		//cudaMemcpy(pop_vals, agent_val, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToHost);
//		//for (int i = 0; i < NUM_OF_AGENTS; ++i)
//		//{
//		//	cout << i << ".    " << pop_vals[i] << ", " << endl;
//		//}
//		//cout << endl;
//	
//		//cudaMemcpy(pop_back, ff_new_poss, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToHost);
//		//for (int i = 0; i < NUM_OF_AGENTS; ++i)
//		//{
//		//	cout << i << " ";
//		//	for (int j = 0; j < NUM_OF_DIMS; ++j)
//		//	{
//		//		cout << pop_back[i * NUM_OF_DIMS + j] << ", ";
//		//	}
//		//	cout << '\n' << endl;
//		//}
//		//cout << '\n' << endl;
//	
//		searchForBestKernel << <1, NUM_OF_AGENTS_HALF >> > (agent_val, indice);
//	
//		cudaMemcpy(ind, indice, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//		cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
//		error_h(cudaGetLastError());
//	}
//
//	//FF end
//
//	//PSO start
//	
//	//float* r;
//	//unsigned int num_of_ri = 2 * NUM_OF_INDICES;
//	//curandGenerator_t r_int;
//	//cudaMalloc(&r, num_of_ri * sizeof(float));
//	//
//	//curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
//	//curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
//	//
//	//agent_best_pso = agent_pos;
//	//agent_best_pso_v = agent_val;
//	//for (int i = 0; i < MAX_ITER; ++i)
//	//{
//	//	curandGenerateUniform(r_int, r, num_of_ri);
//	//	pso_f << <  NUM_OF_AGENTS, NUM_OF_DIMS>> > (0.1, 0.25, 2, a, b, r, indice, agent_pos, agent_best_pso, agent_val);
//	//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (agent_pos, agent_val);
//	//	compare_two_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_best_pso, agent_best_pso_v, agent_pos, agent_val);
//	//	cudaDeviceSynchronize();
//	//	searchForBestKernel << <1, NUM_OF_AGENTS_HALF >> > (agent_best_pso_v, indice);
//	//	cudaMemcpy(ind, indice, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//	//	err = cudaMemcpy(&best[i], &agent_best_pso_v[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
//	//	error_h(cudaGetLastError());
//	//}
//	
//	//PSO end
//
//
//
//	//iGWO start
///*
//	dim3 agents(NUM_OF_AGENTS, NUM_OF_AGENTS, 1);
//
//	float* r_a; float* r_d; float* r;
//	unsigned int* r_w; unsigned int* r_nh;
//	curandGenerator_t r_in;
//	unsigned int num_of_uR = pow(NUM_OF_AGENTS, 2);
//	cudaMalloc(&r_a, 3 * NUM_OF_AGENTS * sizeof(float));
//	cudaMalloc(&r_d, 3 * NUM_OF_AGENTS * sizeof(float));
//	cudaMalloc(&r, NUM_OF_INDICES * sizeof(float));
//
//	cudaMalloc(&r_w, NUM_OF_AGENTS* sizeof(unsigned int));
//	cudaMalloc(&r_nh, NUM_OF_AGENTS * sizeof(unsigned int));
//
//	curandCreateGenerator(&r_in, CURAND_RNG_PSEUDO_PHILOX4_32_10);
//	curandSetPseudoRandomGeneratorSeed(r_in, time(NULL));
//
//	curandGenerateUniform(r_in, r_a, 6 * NUM_OF_AGENTS);
//	curandGenerateUniform(r_in, r, NUM_OF_INDICES);
//	curandGenerate(r_in, r_w, 2 * NUM_OF_AGENTS);
//
//	cudaError_t eer;
//	float A = 0;
//	double aaa;
//	auto s = std::chrono::high_resolution_clock::now();;
//	long long e = 0; long long ee = 0; long long eee = 0; long long eeee = 0; long long eeeee = 0; long long eew = 0; long long eeew = 0;
//	for (int i = 0; i < MAX_ITER; ++i) 
//	{
//		A = 2 - 2 * i / MAX_ITER;
//
//		searchForBestThree << <1, NUM_OF_AGENTS_HALF >> > (agent_val, indice);
//
//		//cudaMemcpy(indi, indice, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//
//		//err = cudaMemcpy(&best[i], &agent_val[indi[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
//
//		GWO << < NUM_OF_AGENTS, NUM_OF_DIMS>> > (indice, r_a, a, b, A, agent_pos, X_gwo);
//
//		calc_distances<<<agents,1>>>(agent_pos, dist_gwo);
//
//		iGWO_nh << <agents, NUM_OF_DIMS>> > (r_w, r, a, b, dist_gwo, agent_pos, X_gwo, nh_pos, ind_to_choose);
//
//		cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (nh_pos, nh_val);
//	
//		cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF>> > (X_gwo, agent_val);
//		
//		eer = cudaGetLastError();
//
//		//cudaDeviceSynchronize();
//		compare_two_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, X_gwo, agent_val, nh_pos, nh_val);
//		curandGenerateUniform(r_in, r_a, 6 * NUM_OF_AGENTS);
//		curandGenerateUniform(r_in, r, NUM_OF_INDICES);
//		curandGenerate(r_in, r_w, 2 * NUM_OF_AGENTS);
//
//
//		cudaMemcpy(indi, dist_gwo, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//			for (int i = 0; i <NUM_OF_AGENTS; ++i)
//			{
//				cout << indi[i] << ", " << endl;
//			}
//		cout << '\n' << endl;
//	}*/
//	/*	s = std::chrono::high_resolution_clock::now();
//		eeew += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();
//	e /= 1000; ee /= 1000; eee /= 1000; eeee /= 1000; eeeee /= 1000; eeew /= 1000; eew /= 1000;*/
//	//igwo end
//
//	//abc start
//
//	//float* r;
//	//unsigned int* rI;
//	//unsigned int* abbadon_dec;
//	//unsigned int* index_to_rns;
//	//curandGenerator_t r_int;
//	//cudaMalloc(&r, NUM_OF_INDICES * sizeof(float));
//	//cudaMalloc(&rI, NUM_OF_AGENTS * sizeof(float));
//	//cudaMalloc(&index_to_rns, NUM_OF_AGENTS * sizeof(unsigned int));
//	//cudaMalloc(&abbadon_dec, NUM_OF_AGENTS * sizeof(unsigned int));
//	//curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_MTGP32);
//	//curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));
//	//
//
//	//unsigned int abbadon_val = MAX_ITER / 2;
//
//	//for (int i = 0; i < MAX_ITER; ++i)
//	//{
//	//	curandSetPseudoRandomGeneratorSeed(r_int, time(NULL) + i + 3);
//	//	curandGenerate(r_int, rI, NUM_OF_AGENTS);
//	//	curandGenerateUniform(r_int, r, NUM_OF_INDICES);
//
//	//	abc_rns << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_best_pso, a, b, r, rI);
//	//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (agent_best_pso, agent_best_pso_v, cost_func_tmp);
//	//	compare_two_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, agent_best_pso, agent_best_pso_v, abbadon_dec);
//	//	probability_selection << <1, NUM_OF_AGENTS >> > (agent_val, r, index_to_rns);
//
//	//	curandGenerate(r_int, rI, NUM_OF_AGENTS);
//	//	curandGenerateUniform(r_int, r, NUM_OF_INDICES);
//
//	//	?????
//	//	abc_rns << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_best_pso, index_to_rns, a, b, r, rI);
//	//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (agent_best_pso, agent_best_pso_v, cost_func_tmp);
//	//	compare_two_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, agent_best_pso, agent_best_pso_v, abbadon_dec);
//
//	//	curandGenerateUniform(r_int, r, NUM_OF_INDICES);
//	//	err = cudaDeviceSynchronize();
//
//	//	scout_phase<<<NUM_OF_AGENTS, NUM_OF_DIMS>>>(abbadon_dec, MAX_ITER/2, a, b, r, agent_pos, ind[0]);
//	//	cost_func << <NUM_OF_AGENTS, DIMS_TO_LOG_HALF >> > (agent_pos, agent_val, cost_func_tmp);
//	//	searchForBestKernel << <1, NUM_OF_AGENTS_HALF>> > (agent_val, indice);
//
//	//	err2 = cudaMemcpy(pop_vals, agent_val, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToHost);
//	//	err2 = cudaMemcpy(pop_back, agent_best_pso_v, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToHost);
//	//	for (int j = 0; j < NUM_OF_AGENTS; ++j)
//	//	{
//	//		if (pop_back[j] == pop_vals[j])
//	//			cout << j << "     " << pop_back[j] << "........" << pop_vals[j] << endl;
//	//	}
//	//	cout << endl;
//
//
//	//	err2 = cudaMemcpy(ind, indice, NUM_OF_AGENTS * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//	//	err1 = cudaMemcpy(&best[i], &agent_val[ind[0]], sizeof(float), ::cudaMemcpyDeviceToHost);
//
//	//	error_h(cudaGetLastError());
//
//	//}
//	
//	err = cudaMemcpy(pop_back, agent_pos, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToHost);
//	err = cudaMemcpy(pop_vals, agent_val, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToHost);
//
//	cout << ind[0] << ", " << endl;
//	for (int i = 0; i < MAX_ITER; ++i)
//	{
//		cout << ind[0] << ".   " << best[i] << "---- " << endl;
//		//for (int j = 0; j < NUM_OF_DIMS; ++j)
//		//{
//		//	cout << pop_back[ind[0] * NUM_OF_DIMS + j] << ", ";
//		//}
//		//	cout << endl;
//	}
//	cout << '\n' << endl;
//	cout << ind[0] << ".   " << pop_vals[ind[0]] << "---- " << endl;
//
//
//	free(pop_back);
//	free(pop_vals);
//	free(best);
//
//	cudaFree(agent_pos);
//	cudaFree(agent_val);
//	cudaFree(indice);
//	cudaFree(a);
//	cudaFree(b);
//	cudaFree(best_sol_a);
//	cudaFree(y_DE);
//	cudaFree(y_DE_val);
//	cudaFree(best_DE);
//	cudaFree(best_de);
//
//
//
//	return 0;
//}
//
//
////curand array
////float* r = NULL;
////cudaMalloc(&r, NUM_OF_INDICES * sizeof(float));
////curandCreateGenerator(&pseudo_rand, CURAND_RNG_PSEUDO_PHILOX4_32_10);
////curandSetPseudoRandomGeneratorSeed(pseudo_rand, 1);
////curandGenerateUniform(pseudo_rand, r, NUM_OF_INDICES);
////curandGenerateUniform(pseudo_rand, r, NUM_OF_INDICES);
////curandGenerateUniform(pseudo_rand, r, NUM_OF_INDICES);
//
//
//	//err = cudaMemcpy(pop_back, agent_pos, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToHost);
//	//for (int i = 0; i < NUM_OF_AGENTS; ++i)
//	//{
//	//	cout << i << " ";
//	//	for (int j = 0; j < NUM_OF_DIMS; ++j)
//	//	{
//	//		cout << pop_back[i * NUM_OF_DIMS + j] << ", ";
//	//	}
//	//	cout << '\n' << endl;
//	//}
//	//cout << '\n' << endl;
//	
//		//err2 = cudaMemcpy(pop_vals, agent_pos, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToHost);
//		//for (int i = 0; i < NUM_OF_AGENTS; ++i)
//		//{
//		//	cout << pop_vals[i] << ", " << endl;
//		//}
//		
//	//int* gg;
//	//cudaMalloc(&gg, 4 * sizeof(int));
//	//int *gge = (int*)malloc(4* sizeof(int));
//	//tet << <agents, block>> > (gg);
//	//err =cudaMemcpy(gge, gg, 4 * sizeof(int), ::cudaMemcpyDeviceToHost);
//	//for (int i = 0; i < 4; ++i)
//	//{
//	//	cout << gge[i] << endl;
//	//}

