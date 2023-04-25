
#include"alg.cuh"


 __host__ void Diff_ev(const float w, const float p, const float* init_pop,const float* init_vals,const int* a,const  int* b,
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;			
	float* agent_val = NULL;
	float* y_DE_pos = NULL;
	float* y_DE_val = NULL;

	unsigned int* best_index = NULL;
	unsigned int* r = NULL;
	unsigned int* X = NULL;
	unsigned int* HOST_best_index = (unsigned int*)malloc(num_of_agents_half * sizeof(unsigned int)); 
	float* Rj;


	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_int;
	unsigned int num_of_Ri = 4 * num_of_indices;
	
	error_h(cudaGetLastError());

	curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandStatus_t aaa =  curandSetPseudoRandomGeneratorSeed(r_int, 5);

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
	cudaMalloc(&y_DE_pos, num_of_indices * sizeof(float));
	cudaMalloc(&y_DE_val, num_of_agents * sizeof(float));
	cudaMalloc(&r, num_of_Ri * sizeof(unsigned int));
	cudaMalloc(&X, num_of_indices * sizeof(unsigned int));
	cudaMalloc(&Rj, num_of_indices * sizeof(float));
	cudaMalloc(&best_index, num_of_agents_half* sizeof(unsigned int));

	err = cudaGetLastError();

	cudaMemcpy(agent_pos, init_pop, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_val, init_vals, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToDevice);
	
	auto s = std::chrono::high_resolution_clock::now();
	long long iter_time = 0;

	for (int i = 0; i < max_iter; ++i)
	{
		s = std::chrono::high_resolution_clock::now();
		
		//init rands
		curandGenerate(r_int, r, num_of_Ri);
		curandGenerate(r_int, X, num_of_indices);
		curandGenerateUniform(r_int, Rj, num_of_indices);
		
		//calc new pos
		DE <<<num_of_agents, num_of_dims>>> (w, p, a, b, r, X, Rj, best_index, agent_pos, agent_val, y_DE_pos);
		cost_func <<<num_of_agents, dims_to_log_half >>> (y_DE_pos, y_DE_val);
		compare_two_pop <<<num_of_agents, num_of_dims >>> (agent_pos, agent_val, y_DE_pos, y_DE_val);

		//find best pos
		cudaDeviceSynchronize();
		searchForBestKernel <<<1, num_of_agents_half>>> (agent_val, best_index);
	
		iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();
		
		cudaMemcpy(HOST_best_index, best_index, num_of_agents_half * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_vals[i], &agent_val[HOST_best_index[0]], sizeof(float), ::cudaMemcpyDeviceToDevice);
		
		err = cudaGetLastError();
		
		if (err != 0)
			break;
	}

	cudaMemcpy(best_pos, &agent_pos[HOST_best_index[0]], num_of_dims * sizeof(float), ::cudaMemcpyDeviceToDevice);
	time_per_iter = iter_time / max_iter;

	cudaFree(agent_pos);
	cudaFree(agent_val);
	cudaFree(y_DE_pos);
	cudaFree(y_DE_val);
	cudaFree(r);
	cudaFree(X);
	cudaFree(Rj);

	error_h(err);
}

__host__ void PSO(const float w, const float c1, const float c2, const float* init_pop, const float* init_vals, const int* a, const int* b,
	float* best_pos, float* best_vals, float time_per_iter) {
	
	float* agent_pos = NULL;
	float* agent_best_pos = NULL;
	float* agent_val = NULL;
	float* agent_best_val = NULL;

	unsigned int* best_index = NULL;
	float* r;
	unsigned int num_of_ri = 2 * num_of_indices;
	unsigned int* HOST_best_index = (unsigned int*)malloc(num_of_agents_half * sizeof(unsigned int));


	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_best_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
	cudaMalloc(&agent_best_val, num_of_agents * sizeof(float));
	cudaMalloc(&r, num_of_ri * sizeof(float));
	cudaMalloc(&best_index, num_of_agents_half * sizeof(unsigned int));

	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_int;
	curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_int, time(NULL));

	cudaMemcpy(agent_pos, init_pop, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_best_pos, init_pop, num_of_indices * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_val, init_vals, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_best_val, init_vals, num_of_agents * sizeof(float), ::cudaMemcpyDeviceToDevice);

	auto s = std::chrono::high_resolution_clock::now();
	long long iter_time = 0;

	for (int i = 0; i < max_iter; ++i)
	{
		s = std::chrono::high_resolution_clock::now();

		//init rand
		curandGenerateUniform(r_int, r, num_of_ri);
		
		//calc new pos
		pso_f << <  num_of_agents, num_of_dims >> > (w, c1, c2, a, b, r, best_index, agent_pos, agent_best_pos, agent_val); //V --
		cost_func << <num_of_agents, dims_to_log_half >> > (agent_pos, agent_val);
		compare_two_pop << <num_of_agents, num_of_dims >> > (agent_best_pos, agent_best_val, agent_pos, agent_val);

		//find best pos
		cudaDeviceSynchronize();
		searchForBestKernel << <1, num_of_agents_half >> > (agent_best_val, best_index);
		
		iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();

		cudaMemcpy(HOST_best_index, best_index, num_of_agents_half * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_vals[i], &agent_best_val[HOST_best_index[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

		err = cudaGetLastError();

		if (err != 0)
			break;
	}

	cudaMemcpy(best_pos, &agent_pos[HOST_best_index[0]], num_of_dims * sizeof(float), ::cudaMemcpyDeviceToDevice);
	time_per_iter = iter_time / max_iter;

	cudaFree(agent_pos);
	cudaFree(agent_best_pos);
	cudaFree(agent_val);
	cudaFree(agent_best_val);
	cudaFree(r);
	cudaFree(best_index);

	error_h(err);
}

__host__ void FF(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
}

__host__ void ABC(const float* init_pop,const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));

}

__host__ void GWO(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));

}

__host__ void iGWO(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));

}