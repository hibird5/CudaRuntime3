
#include"alg.cuh"


 __host__ void Diff_ev(const float w, const float p, const float* init_pop,const float* init_vals,const int* a,const  int* b,
	float* best_pos, float* best_vals, float time_per_iter) {
	
	//Initialization
	float* agent_pos = NULL;			
	float* agent_val = NULL;
	float* y_DE_pos = NULL;
	float* y_DE_val = NULL;

	unsigned int* best_index = NULL;
	unsigned int* r = NULL;
	unsigned int* X = NULL;
	unsigned int* HOST_best_index = (unsigned int*)malloc(NUM_OF_AGENTS_HALF * sizeof(unsigned int)); 
	float* Rj;

	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_gen;
	unsigned int num_of_Ri = 4 * NUM_OF_INDICES;

	curandCreateGenerator(&r_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_gen, 5);

	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&y_DE_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&y_DE_val, NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&r, num_of_Ri * sizeof(unsigned int));
	cudaMalloc(&X, NUM_OF_INDICES * sizeof(unsigned int));
	cudaMalloc(&Rj, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&best_index, NUM_OF_AGENTS_HALF* sizeof(unsigned int));

	err = cudaGetLastError();

	cudaMemcpy(agent_pos, init_pop, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_val, init_vals, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToDevice);
	
	error_h(cudaGetLastError());

	auto s = std::chrono::high_resolution_clock::now();
	long long iter_time = 0;

	//Loop
	for (int i = 0; i < MAX_ITER; ++i)
	{
		s = std::chrono::high_resolution_clock::now();
		
		//init rands
		curandGenerate(r_gen, r, num_of_Ri);
		curandGenerate(r_gen, X, NUM_OF_INDICES);
		curandGenerateUniform(r_gen, Rj, NUM_OF_INDICES);
		
		//calc new pos
		DE <<<NUM_OF_AGENTS, NUM_OF_DIMS>>> (w, p, a, b, r, X, Rj, best_index, agent_pos, agent_val, y_DE_pos);
		cost_func <<<NUM_OF_AGENTS, DIMS_TO_LOG_HALF >>> (y_DE_pos, y_DE_val);
		compare_two_pop <<<NUM_OF_AGENTS, NUM_OF_DIMS >>> (agent_pos, agent_val, y_DE_pos, y_DE_val);

		//find best pos
		cudaDeviceSynchronize();
		searchForBestKernel <<<1, NUM_OF_AGENTS_HALF>>> (agent_val, best_index);
	
		iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();
		
		cudaMemcpy(HOST_best_index, best_index, NUM_OF_AGENTS_HALF * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_vals[i], &agent_val[HOST_best_index[0]], sizeof(float), ::cudaMemcpyDeviceToDevice);
		
		err = cudaGetLastError();
		
		if (err != 0)
			break;
	}

	cudaMemcpy(best_pos, &agent_pos[HOST_best_index[0]], NUM_OF_DIMS * sizeof(float), ::cudaMemcpyDeviceToDevice);
	time_per_iter = iter_time / MAX_ITER;

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
	
	//Initialization
	float* agent_pos = NULL;
	float* agent_best_pos = NULL;
	float* agent_val = NULL;
	float* agent_best_val = NULL;

	unsigned int* best_index = NULL;
	float* r;
	unsigned int num_of_ri = 2 * NUM_OF_INDICES;
	unsigned int* HOST_best_index = (unsigned int*)malloc(NUM_OF_AGENTS_HALF * sizeof(unsigned int));

	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_gen;
	curandCreateGenerator(&r_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_gen, time(NULL));

	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_best_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&agent_best_val, NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&r, num_of_ri * sizeof(float));
	cudaMalloc(&best_index, NUM_OF_AGENTS_HALF * sizeof(unsigned int));


	cudaMemcpy(agent_pos, init_pop, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_best_pos, init_pop, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_val, init_vals, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_best_val, init_vals, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToDevice);


	auto s = std::chrono::high_resolution_clock::now();
	long long iter_time = 0;

	//Loop
	for (int i = 0; i < MAX_ITER; ++i)
	{
		s = std::chrono::high_resolution_clock::now();

		//init rand
		curandGenerateUniform(r_gen, r, num_of_ri);
		
		//calc new pos
		pso_f <<<NUM_OF_AGENTS, NUM_OF_DIMS >>> (w, c1, c2, a, b, r, best_index, agent_pos, agent_best_pos, agent_val); //V --
		cost_func <<<NUM_OF_AGENTS, DIMS_TO_LOG_HALF >>> (agent_pos, agent_val);
		compare_two_pop <<<NUM_OF_AGENTS, NUM_OF_DIMS >>> (agent_best_pos, agent_best_val, agent_pos, agent_val);

		//find best pos
		cudaDeviceSynchronize();
		searchForBestKernel << <1, NUM_OF_AGENTS_HALF >> > (agent_best_val, best_index);
		
		iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();

		cudaMemcpy(HOST_best_index, best_index, NUM_OF_AGENTS_HALF * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_vals[i], &agent_best_val[HOST_best_index[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

		err = cudaGetLastError();

		if (err != 0)
			break;
	}

	cudaMemcpy(best_pos, &agent_pos[HOST_best_index[0]], NUM_OF_DIMS * sizeof(float), ::cudaMemcpyDeviceToDevice);
	time_per_iter = iter_time / MAX_ITER;

	cudaFree(agent_pos);
	cudaFree(agent_best_pos);
	cudaFree(agent_val);
	cudaFree(agent_best_val);
	cudaFree(r);
	cudaFree(best_index);

	error_h(err);
}

__host__ void FF(const float alfa, const float beta, const float gamma, const float* init_pop, const float* init_vals, const int* a, const int* b,
	float* best_pos, float* best_vals, float time_per_iter) {

	//Initialization
	float* agent_pos = NULL;
	float* agent_val = NULL;
	float* new_pos = NULL;
	float* new_vals = NULL;
	float* cost_func_tmp = NULL;

	unsigned int* best_index = NULL;
	unsigned int* HOST_best_index = (unsigned int*)malloc(NUM_OF_AGENTS_HALF * sizeof(unsigned int));
	float* r;
	unsigned int num_of_uR = POW_OF_AGENTS * NUM_OF_DIMS;

	dim3 agents(NUM_OF_AGENTS, NUM_OF_AGENTS, 1);
	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_gen;
	curandCreateGenerator(&r_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(r_gen, time(NULL));
	
	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&best_index, NUM_OF_AGENTS_HALF * sizeof(unsigned int));
	cudaMalloc(&r, num_of_uR * sizeof(float));
	cudaMalloc(&new_pos, NUM_OF_AGENTS * NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&new_vals, NUM_OF_AGENTS * NUM_OF_AGENTS * sizeof(float));
	cudaMalloc(&cost_func_tmp, POW_OF_AGENTS * DIMS_TO_LOG_HALF * sizeof(float));

	cudaMemcpy(agent_pos, init_pop, NUM_OF_INDICES * sizeof(float), ::cudaMemcpyDeviceToDevice);
	cudaMemcpy(agent_val, init_vals, NUM_OF_AGENTS * sizeof(float), ::cudaMemcpyDeviceToDevice);

	auto s = std::chrono::high_resolution_clock::now();
	long long iter_time = 0;

	for (int i = 0; i < MAX_ITER; ++i)
	{
		s = std::chrono::high_resolution_clock::now();
		//init rand
		curandGenerateNormal(r_gen, r, num_of_uR, 0.0, 0.5);

		//calc new pos
		ffa << <agents, NUM_OF_DIMS >> > (alfa, beta, gamma, a, b, r, agent_pos, new_pos, agent_val);
		cost_func << <agents, DIMS_TO_LOG_HALF >> > (new_pos, new_vals, cost_func_tmp);
		compare_ff_pop << <NUM_OF_AGENTS, NUM_OF_DIMS >> > (agent_pos, agent_val, new_pos, new_vals);

		//find best sol
		cudaDeviceSynchronize();
		searchForBestKernel << <1, NUM_OF_AGENTS_HALF >> > (agent_val, best_index);
		
		iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();

		cudaMemcpy(HOST_best_index, best_index, NUM_OF_AGENTS_HALF * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_vals[i], &agent_val[HOST_best_index[0]], sizeof(float), ::cudaMemcpyDeviceToHost);

		err = cudaGetLastError();

		if (err != 0)
			break;

	}

	cudaMemcpy(best_pos, &agent_pos[HOST_best_index[0]], NUM_OF_DIMS * sizeof(float), ::cudaMemcpyDeviceToDevice);
	time_per_iter = iter_time / MAX_ITER;

	cudaFree(agent_pos);
	cudaFree(agent_val);
	cudaFree(best_index);
	cudaFree(r);
	cudaFree(new_pos);
	cudaFree(new_vals);
	cudaFree(cost_func_tmp);

	error_h(err);
}

__host__ void ABC(const float* init_pop,const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));

}

__host__ void GWO(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));

}

__host__ void iGWO(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, NUM_OF_INDICES * sizeof(float));
	cudaMalloc(&agent_val, NUM_OF_AGENTS * sizeof(float));

}