#include"alg.cuh"

__host__  void Diff_ev(const float w, const float p, const float* init_pop,const float* init_vals,const int* a,const  int* b,
	float* best_pos, float* best_vals, float time_per_iter) {

	float* agent_pos = NULL;
	float* agent_val = NULL;
	float* y_DE_pos = NULL;
	float* y_DE_val = NULL;

	unsigned int* best_index = NULL;
	unsigned int* r;
	unsigned int* X;
	unsigned int* best_ind = (unsigned int*)malloc(num_of_agents_half * sizeof(unsigned int)); ;
	float* Rj;


	cudaError_t err = (cudaError_t)0;
	curandGenerator_t r_int;
	unsigned int num_of_Ri = 4 * num_of_indices;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
	cudaMalloc(&y_DE_pos, num_of_indices * sizeof(float));
	cudaMalloc(&y_DE_val, num_of_agents * sizeof(float));
	cudaMalloc(&r, num_of_Ri * sizeof(unsigned int));
	cudaMalloc(&X, num_of_indices * sizeof(unsigned int));
	cudaMalloc(&Rj, num_of_indices * sizeof(float));
	cudaMalloc(&best_index, num_of_agents_half* sizeof(unsigned int));

	curandCreateGenerator(&r_int, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	err = cudaGetLastError();
	curandStatus_t aaa =  curandSetPseudoRandomGeneratorSeed(r_int, 5);
	//err = cudaGetLastError();
	
	//auto s = 0; // std::chrono::high_resolution_clock::now();
	//long long iter_time ;

	for (int i = 0; i < max_iter; ++i)
	{
		//s = std::chrono::high_resolution_clock::now();
		
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
	
		
		err = cudaGetLastError();
		cudaMemcpy(&best_vals[i], &agent_val[best_index[0]], sizeof(float), ::cudaMemcpyDeviceToDevice);
		//iter_time += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - s).count();
		if (err != 0)
			break;
	}

	cudaMemcpy(best_pos, &agent_pos[best_index[0]], num_of_dims * sizeof(float), ::cudaMemcpyDeviceToDevice);
	//time_per_iter = iter_time / max_iter;

	cudaFree(agent_pos);
	cudaFree(agent_val);
	cudaFree(y_DE_pos);
	cudaFree(y_DE_val);
	cudaFree(r);
	cudaFree(X);
	cudaFree(Rj);

	error_h(err);
}

__host__ void PSO(const float* init_pop, const float* init_vals, const int* a, const int* b, 
	float* best_pos, float* best_vals, float time_per_iter) {
	
	float* agent_pos = NULL;
	float* agent_val = NULL;

	cudaMalloc(&agent_pos, num_of_indices * sizeof(float));
	cudaMalloc(&agent_val, num_of_agents * sizeof(float));
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