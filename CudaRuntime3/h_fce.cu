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
	int agent = blockIdx.x * num_of_dims;
	agent_val[blockIdx.x] = 0;
	switch (input_func)
	{
	case 1:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[blockIdx.x] += pow(agent_pos[agent + i], 4) - 16 * pow(agent_pos[agent + i], 2)
				+ 5 * agent_pos[agent + i];
		}
		agent_val[blockIdx.x] /= 2;
		break;
	case 2:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[blockIdx.x] += 220 / (i + blockIdx.x + 1);
		}
		agent_val[blockIdx.x] /= 2;
		break;

	case 3:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[blockIdx.x] += 220 / (i + blockIdx.x + 1);
		}
		agent_val[blockIdx.x] /= 2;
		break;

	default:
#pragma unroll 
		for (int i = 0; i < num_of_dims; ++i)
		{
			agent_val[blockIdx.x] += pow(agent_pos[agent + i], 2);
		}
		break;
	};
}


__global__ void DE(const float w, const float p, const int* a, const int* b, const unsigned long seed, const size_t* best_sol,
	const float* agent_pos, const float* agent_val, float* y)
{
	float u_tmp = 0;
	float u;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int i_r1, i_r2, i_r3, i_r4, X;
	float Rj;
	curandState r1;
	curand_init(seed, blockIdx.x, 0, &r1);
	i_r1 = (curand(&r1) % num_of_agents) * blockDim.x + threadIdx.x;
	i_r2 = (curand(&r1) % num_of_agents) * blockDim.x + threadIdx.x;
	i_r3 = (curand(&r1) % num_of_agents) * blockDim.x + threadIdx.x;
	i_r4 = (curand(&r1) % num_of_agents) * blockDim.x + threadIdx.x;

	u_tmp = (index < num_of_indices) ?
		//agent_pos[best_sol[0] * blockDim.x + threadIdx.x] + w * (agent_pos[i_r1] + agent_pos[i_r2] - agent_pos[i_r3] - agent_pos[i_r4])
		agent_pos[i_r1] + w * (agent_pos[i_r2] - agent_pos[i_r3])
		:
		u_tmp;

	//search dom test
	u = (a[threadIdx.x] <= u_tmp) ? u_tmp : a[threadIdx.x];
	u = (b[threadIdx.x] >= u_tmp) ? u_tmp : b[threadIdx.x];


	__syncthreads();
	X = curand(&r1) % num_of_dims;
	curand_init(seed, 0, index, &r1);
	Rj = curand_uniform(&r1);

	y[index] = (Rj <= p || X == threadIdx.x) ? u : agent_pos[index] ;
	__syncthreads();
}

__global__ void pso_f(const float w, const float c1, const float c2, const int* a, const int* b, const unsigned long seed,
	const size_t* best_sol, float* agent_pos, const float* agent_best_pos, const float* agent_val)
{
	float V = 0;
	float tmp = 0;
	float r1, r2;
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int best_index = threadIdx.x + best_sol[0] * blockDim.x;
	
	curandState r;
	curand_init(seed, blockIdx.x, 0, &r);
	r1 = curand_uniform(&r);
	r2 = curand_uniform(&r);

	__syncthreads();
	
	V = w * V + c1 * r1*(agent_best_pos[index] - agent_pos[index]) + c2 * r2*(agent_best_pos[best_index] - agent_pos[index]);

	tmp = V + agent_pos[index];

	agent_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	agent_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];

	__syncthreads();
}

__global__ void compare_two_pop(float* f_pos, float* f_val, const float* s_pos, const float* s_val)
{
	//f_val[blockIdx.x] = (s_val[blockIdx.x] < f_val[blockIdx.x]) ? s_val[blockIdx.x] : f_val[blockIdx.x];
	unsigned int ind;

	if (s_val[blockIdx.x] < f_val[blockIdx.x])
	{
		f_val[blockIdx.x] = s_val[blockIdx.x];
		__syncthreads();
			ind = blockIdx.x * num_of_dims;
#pragma unroll
		for (int i = ind; i < ind + num_of_dims; ++i)
		{
			f_pos[i] = s_pos[i];
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