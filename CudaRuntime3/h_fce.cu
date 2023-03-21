#include"h_fce.cuh"

__global__ void get_constr(const int min, const int max, int* a, int* b)
{
	if (!a || !b)
		return;

	a[blockIdx.x] = min;
	b[blockIdx.x] = max;
	
	return;
}

__global__ void init_pop_pos(float* agent_pos, const int* a, const int* b,unsigned long seed)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init(seed, index, 0, &state);

	agent_pos[index] = (index < num_of_indices) ?
		a[threadIdx.x] + curand_uniform(&state) * (b[threadIdx.x] - a[threadIdx.x]) : 0;

}

__global__ void cost_func(const float* agent_pos, float* agent_val)
{
	unsigned int agent = threadIdx.x * num_of_dims + threadIdx.y * num_of_indices;
	unsigned int offset= threadIdx.x + threadIdx.y * num_of_agents;
	agent_val[threadIdx.x] = 0;
	switch (input_func)
	{
	case 1:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[offset] += powf(agent_pos[agent + i], 4) - 16 * powf(agent_pos[agent + i], 2)
				+ 5 * agent_pos[agent + i];
		}
		agent_val[offset] /= 2;
		break;

	default:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[offset] += powf(agent_pos[agent + i], 2);
		}
		break;
	}
}
//__global__ void sphere(const float* agent_pos, float* agent_val)
//{
//	int agent = threadIdx.x * num_of_dims;
//	agent_val[threadIdx.x] = 0;
//	
//#pragma unroll 
//		for (int i = 0; i < num_of_dims; ++i)
//		{
//			agent_val[threadIdx.x] += pow(agent_pos[agent + i], 2);
//		}
//
//}
//__global__ void styblinski�tang(const float* agent_pos, float* agent_val)
//{
//	int agent = threadIdx.x * num_of_dims;
//	agent_val[threadIdx.x] = 0;
//
//#pragma unroll 
//	for (int i = 0; i < num_of_dims; ++i)
//	{
//		agent_val[threadIdx.x] += pow(agent_pos[agent + i], 4) - 16 * pow(agent_pos[agent + i], 2)
//			+ 5 * agent_pos[agent + i];
//	}
//	agent_val[threadIdx.x] /= 2;
//}

__global__ void DE(const float w, const float p, const int* a, const int* b, 
	const unsigned int* Ri, const unsigned int* X, const float* Rj,
	const size_t* best_sol, const float* agent_pos, const float* agent_val, float* y)
{
	float u_tmp = 0;
	float u;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int r_index = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	unsigned int i_r1, i_r2, i_r3, i_r4;

	i_r1 = (Ri[r_index + 0] % num_of_agents) * blockDim.x + threadIdx.x;
	i_r2 = (Ri[r_index + 1] % num_of_agents) * blockDim.x + threadIdx.x;
	i_r3 = (Ri[r_index + 2] % num_of_agents) * blockDim.x + threadIdx.x;
	i_r4 = (Ri[r_index + 3] % num_of_agents) * blockDim.x + threadIdx.x;

	u_tmp = (index < num_of_indices) ?
		agent_pos[best_sol[0] * blockDim.x + threadIdx.x] + w * (agent_pos[i_r1] + agent_pos[i_r2] - agent_pos[i_r3] - agent_pos[i_r4])
		//agent_pos[i_r1] + w * (agent_pos[i_r2] - agent_pos[i_r3])
		:
		u_tmp;

	//search dom test
	u = (a[threadIdx.x] <= u_tmp) ? u_tmp : a[threadIdx.x];
	u = (b[threadIdx.x] >= u_tmp) ? u_tmp : b[threadIdx.x];

	//new pos
	y[index] = (Rj[index] <= p || X[index] == threadIdx.x) ? u : agent_pos[index] ;
	//__syncthreads();
}

__global__ void pso_f(const float w, const float c1, const float c2, const int* a, const int* b, const float* r_i,
	const size_t* best_sol, float* agent_pos, const float* agent_best_pos, const float* agent_val)
{
	float V = 0;
	float tmp = 0;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int best_index = threadIdx.x + best_sol[0] * blockDim.x;
	
	unsigned int r2 = index + num_of_indices;
	
	V = w * V + c1 * r_i[index]*(agent_best_pos[index] - agent_pos[index]) + c2 * r_i[r2] 
		* (agent_best_pos[best_index] - agent_pos[index]);

	tmp = V + agent_pos[index];

	agent_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	agent_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];

	__syncthreads();
}


__global__ void ffa(const float alfa, const float beta, const float gamma, const int* a, const int* b, const float* r,
	 const float* agent_pos, float* agent_new_pos, const float* agent_val)
{
	unsigned int agent_x = blockIdx.x;
	unsigned int agent_y = blockIdx.y;

	unsigned int offset_x = blockIdx.x * blockDim.x;
	unsigned int offset_y = blockIdx.y * blockDim.x;

	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index_y = blockIdx.y * blockDim.x + threadIdx.x;
	unsigned int index_r = agent_x + agent_y * num_of_agents;
	unsigned int index	= index_x + blockDim.x * num_of_agents * agent_y;	// index of pos to save

	float R;
	float tmp;

	if (agent_val[agent_y] < agent_val[agent_x]) {
		R = 0;
#pragma unroll
		for (auto i = 0; i < num_of_dims; i++) {
			R += powf(agent_pos[offset_x + i] - agent_pos[offset_y + i], 2);	//calc distance
		}

		tmp = agent_pos[index_x] + beta * exp(-gamma * R) * (agent_pos[index_x] - agent_pos[index_y])
			+ alfa * r[index_r];		//new pos

		agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
		agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
	}
	else {
			agent_new_pos[index] = agent_pos[index_x];		//save old 
	}
	__syncthreads();
}

__global__ void compare_two_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val)
{
	unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	old_val[blockIdx.x] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_val[blockIdx.x] : old_val[blockIdx.x];
	old_pos[ind] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_pos[ind] : old_pos[ind];
	//if (new_val[blockIdx.x] < old_val[blockIdx.x])
	//{
	//	old_val[blockIdx.x] = new_val[blockIdx.x];
	//	
	//	old_pos[ind] = new_pos[ind];
	//}

}

__global__ void compare_ff_pos(float* old_pos, float* old_val, const float* new_pos, const float* new_val)
{
	unsigned int old_index, new_index;
	unsigned int max_agent = num_of_agents * (num_of_agents-1);

#pragma unroll
	for (auto i = blockIdx.x; i < blockIdx.x + max_agent; i += num_of_agents) {
		if (new_val[i] < old_val[blockIdx.x])
		{
			old_val[blockIdx.x] = new_val[i];
			__syncthreads();
			old_index = blockIdx.x * num_of_dims;
			new_index = blockIdx.x * num_of_indices;
#pragma unroll
			for (auto i = 0; i < num_of_dims; ++i)
			{
				old_pos[i + old_index] = new_pos[i + new_index];
			}
		}
	}
		__syncthreads();
}


//__global__ void best_sol(const int num_of_agents, const float* agent_val, size_t* indice, float* best_val)
//{
//	int j = blockIdx.x * num_of_agents / blockDim.x;
//	best_val[blockIdx.x] = agent_val[j];
//	indice[blockIdx.x] = j;
//
//#pragma unroll
//	for (size_t i = j; i < j + num_of_agents / blockDim.x; ++i)
//	{
//		indice[blockIdx.x] = (agent_val[i] < best_val[blockIdx.x]) ? i : indice[blockIdx.x];
//		best_val[blockIdx.x] = (i == indice[blockIdx.x]) ? agent_val[i] : best_val[blockIdx.x];
//	}
//
//}

	//__global__ void ffa(const float alfa, const float beta, const float gamma, const int* a, const int* b, const unsigned long seed,
	//	const size_t* best_sol, float* agent_pos, float* agent_new_pos, const float* agent_val)
	//{
	//	unsigned int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	//	unsigned int index_y = threadIdx.x + blockIdx.y * blockDim.y;
	//	unsigned int agent_start = blockIdx.x * blockDim.x*sizeof(double);
	//
	//	__shared__ float R_i[num_of_indices];
	//	__shared__ float  R[num_of_agents];
	//	float  tmp;
	//	float  agent_diff = agent_pos[index] - agent_pos[best_sol[0]];
	//	curandState r;
	//	curand_init(seed, index, 0, &r);
	//
	//	R_i[index] = pow(agent_diff, 2);
	//	__syncthreads();
	//	
	//	R[blockIdx.x] = thrust::reduce(thrust::device, R_i + agent_start, R_i + agent_start + blockDim.x*sizeof(double));
	//	__syncthreads();
	//
	//	tmp = agent_pos[index] + beta * exp(-gamma * R[blockIdx.x]) * agent_diff + alfa * curand_normal(&r);
	//
	//	agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	//	agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
	//
	//}

//__global__ void ffa(const float alfa, const float beta, const float gamma, const int* a, const int* b, const unsigned long seed,
//	const float* agent_pos, float* agent_new_pos, const float* agent_val)
//{
//	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	unsigned int agent_x = blockIdx.x;
//	unsigned int agent_y = threadIdx.x;
//
//	unsigned int offset_x = agent_x * num_of_dims;
//	unsigned int offset_y = agent_y * num_of_dims;
//
//	unsigned int offset_Y = agent_y * (num_of_agents)*num_of_dims;
//
//	unsigned int x_indice;
//
//	float R = 0;
//	float  tmp;
//
//	curandState r;
//	curand_init(seed, index, 0, &r);
//
//	if (agent_val[agent_y] < agent_val[agent_x]) {
//		R = 0;
//#pragma unroll
//		for (auto i = 0; i < num_of_dims; i++) {
//			R += pow(agent_pos[offset_x + i] - agent_pos[offset_y + i], 2);	//calc distance
//		}
//
//#pragma unroll
//		for (auto i = 0; i < num_of_dims; i++) {
//
//			x_indice = offset_x + i;
//			tmp = agent_pos[x_indice] + beta * exp(-gamma * R) * (agent_pos[x_indice] - agent_pos[offset_y + i])
//				+ alfa * curand_normal(&r);		//new pos
//
//			index = x_indice + offset_Y;	//possible pos, 1 column for agent
//			agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
//			agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
//		}
//
//	}
//	else {
//		for (auto i = 0; i < num_of_dims; i++) {
//			x_indice = offset_x + i;
//			index = x_indice + offset_Y;
//			agent_new_pos[index] = agent_pos[x_indice];		//save old 
//		}
//
//	}
//	__syncthreads();
//}
