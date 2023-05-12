#include"h_fce.cuh"

__global__ void get_constr(const int min, const int max, int* a, int* b)
{
	a[blockIdx.x] = min;
	b[blockIdx.x] = max;
	
	return;
}

__global__ void init_pop_pos(float* agent_pos, const int* a, const int* b,unsigned long seed)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state;
	curand_init(seed, index, 0, &state);

	agent_pos[index] = (index < NUM_OF_INDICES) ?
		a[threadIdx.x] + curand_uniform(&state) * (b[threadIdx.x] - a[threadIdx.x]) : 0;

}

__global__ void cost_func(const float* agent_pos, float* agent_val)
{
	unsigned int agent = blockIdx.x * NUM_OF_DIMS;
	__shared__ float tmp[NUM_OF_AGENTS*DIMS_TO_LOG];
	unsigned int index = threadIdx.x + agent;
	unsigned int step = DIMS_TO_LOG_HALF;
	unsigned int step_index = index + step;

	agent_val[blockIdx.x] = 0;
	tmp[index] = 0;

	switch (input_func)
	{
	case 1:
		tmp[index] += ((step_index < agent + NUM_OF_DIMS) ?
			(__powf(fabs(agent_pos[index]), 4) - 16 * __powf(fabs(agent_pos[index]), 2) + 5 * agent_pos[index] +
				powf(fabs(agent_pos[step_index]), 4) - 16 * powf(fabs(agent_pos[step_index]), 2) + 5 * agent_pos[step_index]) / 2
			:
			(__powf(fabs(agent_pos[index]), 4) - 16 * __powf(fabs(agent_pos[index]), 2) + 5 * agent_pos[index]) / 2);
		break;

	case 2:
		tmp[index] += ((step_index < agent + NUM_OF_DIMS) ?
			agent_pos[index] * agent_pos[index] + agent_pos[step_index] * agent_pos[step_index]
			:
			agent_pos[index] * agent_pos[index]);
		break;

	default:
		tmp[index] = nanf(0);
		break;

	}
	step >>= 1;
#pragma unroll
	for (auto i = 0; i < NUM_OF_RUNS_ADD; i++) {
		step_index = threadIdx.x + step;
		tmp[index] += ((step_index) < 2*step) ? tmp[agent + step_index] : 0;
		step >>= 1;
		__syncthreads();
	}
	__syncthreads();
	agent_val[blockIdx.x] = tmp[agent];
}

__global__ void cost_func(const float* agent_pos, float* agent_val, float* tmp)
{
	unsigned int agent = blockIdx.x * NUM_OF_DIMS + blockIdx.y * NUM_OF_INDICES;
	//__shared__ float tmp[NUM_OF_AGENTS][DIMS_TO_LOG_HALF];
	unsigned int index = threadIdx.x + agent;
	unsigned int step = DIMS_TO_LOG_HALF;
	unsigned int step_index = index + step;

	//agent_val[blockIdx.x + blockIdx.y * NUM_OF_AGENTS] = 0;
	tmp[index] = 0;

	switch (input_func)
	{
	case 1:
		tmp[index] += ((step_index < agent + NUM_OF_DIMS) ?
			(__powf(fabs(agent_pos[index]), 4) - 16 * __powf(fabs(agent_pos[index]), 2) + 5 * agent_pos[index] +
				powf(fabs(agent_pos[step_index]), 4) - 16 * powf(fabs(agent_pos[step_index]), 2) + 5 * agent_pos[step_index]) / 2
			:
			(__powf(fabs(agent_pos[index]), 4) - 16 * __powf(fabs(agent_pos[index]), 2) + 5 * agent_pos[index]) / 2);
		break;

	case 2:
		tmp[index] += ((step_index < agent + NUM_OF_DIMS) ?
			agent_pos[index] * agent_pos[index] + agent_pos[step_index] * agent_pos[step_index]
			:
			agent_pos[index] * agent_pos[index]);
		break;

	default:
		tmp[index] = nanf(0);
		break;

	}
	step >>= 1;
#pragma unroll
	for (auto i = 0; i < NUM_OF_RUNS_ADD; i++) {
		step_index = threadIdx.x + step;
		tmp[index] += ((step_index) < 2 * step) ? tmp[agent + step_index] : 0; // 2 * step
		step >>= 1;
		__syncthreads();
	}
	agent_val[blockIdx.x + blockIdx.y * NUM_OF_AGENTS] = tmp[agent];
}
//__device__ float sphere(const float& agent_pos)
//{
//	return (__powf(fabs(agent_pos), 4) - 16 * __powf(fabs(agent_pos), 2) + 5 * agent_pos) / 2;
//}
//__device__ float styblinski–tang(const float agent_pos)
//{
//	return agent_pos * agent_pos;
//}

__global__ void searchForBestKernel(volatile float* objectiveValues, unsigned int* indices)
{
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float oVs[NUM_OF_AGENTS];
	__shared__ unsigned int ind[NUM_OF_AGENTS];

	oVs[id] = objectiveValues[id];
	oVs[id + NUM_OF_AGENTS_HALF] = objectiveValues[id + NUM_OF_AGENTS_HALF];
	ind[id] = id;
	ind[id + NUM_OF_AGENTS_HALF] = id + NUM_OF_AGENTS_HALF;
	__syncthreads();
	unsigned int step = NUM_OF_AGENTS_HALF;

#pragma unroll
	for (int i = 0; i < NUM_OF_RUNS; ++i)
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
	best_three[0] = NUM_OF_AGENTS;
	best_three[1] = NUM_OF_AGENTS;

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float oVs[NUM_OF_AGENTS];
	__shared__ unsigned int ind[NUM_OF_AGENTS];

	for (int j = 0; j < 3; j++) {

		oVs[id] = objectiveValues[id];
		oVs[id + NUM_OF_AGENTS_HALF] = objectiveValues[id + NUM_OF_AGENTS_HALF];
		ind[id] = id;
		ind[id + NUM_OF_AGENTS_HALF] = id + NUM_OF_AGENTS_HALF;
		__syncthreads();
		unsigned int step = NUM_OF_AGENTS_HALF;

#pragma unroll
		for (int i = 0; i < NUM_OF_RUNS; ++i)
		{
			unsigned int ind_s = id + step;
			if (ind[id] == best_three[0] || ind[id] == best_three[1])
				ind[id] = ind[ind_s];
			else if (ind[ind_s] == best_three[0] || ind[ind_s] == best_three[1])
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

__global__ void DE(const float w, const float p, const int* a, const int* b, 
	const unsigned int* Ri, const unsigned int* X, const float* Rj,
	const unsigned int* best_sol, const float* agent_pos, const float* agent_val, float* y)
{
	float u_tmp = 0;
	float u;
	unsigned int index = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	unsigned int r_index = 4 * index;
	unsigned int i_r1, i_r2, i_r3, i_r4;

	i_r1 = (Ri[r_index + 0] % NUM_OF_AGENTS) * NUM_OF_DIMS + threadIdx.x;
	i_r2 = (Ri[r_index + 1] % NUM_OF_AGENTS) * NUM_OF_DIMS + threadIdx.x;
	i_r3 = (Ri[r_index + 2] % NUM_OF_AGENTS) * NUM_OF_DIMS + threadIdx.x;
	i_r4 = (Ri[r_index + 3] % NUM_OF_AGENTS) * NUM_OF_DIMS + threadIdx.x;

	u_tmp = (index < NUM_OF_INDICES) ?
		agent_pos[best_sol[0] * blockIdx.x + threadIdx.x] + w * (agent_pos[i_r1] + agent_pos[i_r2] - agent_pos[i_r3] - agent_pos[i_r4])
		//agent_pos[i_r1] + w * (agent_pos[i_r2] - agent_pos[i_r3])
		:
		u_tmp;

	//search dom test
	u = (a[blockIdx.x] <= u_tmp) ? u_tmp : a[blockIdx.x];
	u = (b[blockIdx.x] >= u_tmp) ? u_tmp : b[blockIdx.x];

	//new pos
	y[index] = (Rj[index] <= p || X[index] == blockIdx.x) ? u : agent_pos[index] ;
	//__syncthreads();
}

__global__ void pso_f(const float w, const float c1, const float c2, const int* a, const int* b, const float* r_i,
	const unsigned int* best_sol, float* agent_pos, const float* agent_best_pos, const float* agent_val)
{
	float V = 0;
	float tmp = 0;
	unsigned int index = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	unsigned int best_index = threadIdx.x + best_sol[0] * NUM_OF_DIMS;
	
	unsigned int r2 = index + NUM_OF_INDICES;
	
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
	unsigned int index_r = index_x + agent_y * NUM_OF_INDICES;
	unsigned int index	= index_x + blockDim.x * NUM_OF_AGENTS * agent_y;	// index of pos to save
	
	float R;
	float tmp;

	if (agent_val[agent_y] < agent_val[agent_x]) {
		R = 0;
#pragma unroll
		for (auto i = 0; i < NUM_OF_DIMS; i++) {
			R += __powf(agent_pos[offset_x + i] - agent_pos[offset_y + i], 2);	//calc distance
		}

		tmp = agent_pos[index_x] + beta * __expf(-gamma * R) * (agent_pos[index_x] - agent_pos[index_y])
			+ alfa * r[index_r];		//new pos

		agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
		agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
	}
	else {
			agent_new_pos[index] = agent_pos[index_x];		//save old 
	}
	__syncthreads();
}

__global__ void GWO(const unsigned int* best_ind, const float* r_a, const int* a, const int* b,
	const float A, const float* agent_pos, float* agent_new_pos) 
{
	unsigned int index    = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	
	unsigned int A_index  = threadIdx.x + best_ind[0] * NUM_OF_DIMS;
	unsigned int B_index  = threadIdx.x + best_ind[1] * NUM_OF_DIMS;
	unsigned int G_index  = threadIdx.x + best_ind[2]* NUM_OF_DIMS;

	unsigned int rA_index = blockIdx.x;
	unsigned int rB_index = blockIdx.x + NUM_OF_AGENTS;
	unsigned int rG_index = blockIdx.x + 2 * NUM_OF_AGENTS;
	unsigned int r_a_index= 3 * NUM_OF_AGENTS;

	float a_alfa = 2 * A * r_a[rA_index] - A;
	float a_beta = 2 * A * r_a[rB_index] - A;
	float a_gamma= 2 * A * r_a[rG_index] - A;

	float d_alfa = fabs(2 * r_a[rA_index + r_a_index] * agent_pos[A_index] - agent_pos[index]);
	float d_beta = fabs(2 * r_a[rB_index + r_a_index] * agent_pos[B_index] - agent_pos[index]);
	float d_gamma= fabs(2 * r_a[rG_index + r_a_index] * agent_pos[G_index] - agent_pos[index]);

	float X_alfa = agent_pos[A_index] - a_alfa * d_alfa;
	float X_beta = agent_pos[B_index] - a_beta * d_beta;
	float X_gamma= agent_pos[G_index] - a_gamma * d_gamma;

	float tmp = (X_alfa + X_beta + X_gamma) / 3;

	agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
}

__global__ void iGWO_nh(unsigned int* r_w, const float* r, const int* a, const int* b,
	const float* distance, float* agent_pos, const float* agent_new_pos, float* nh_pos, unsigned int* ind_to_choose)
{
	unsigned int agent = blockIdx.x * NUM_OF_DIMS;
	unsigned int index = 0;
	unsigned int nh_index = 0;
	unsigned int ind_to_comp = blockIdx.x + blockIdx.x * NUM_OF_AGENTS;	// dist of curr agent n X_gwo
	unsigned int dist_y = blockIdx.x + blockIdx.y * NUM_OF_AGENTS;		// dist of other agents n X_gwo
	unsigned int nh = blockIdx.x + NUM_OF_AGENTS;

	r_w[blockIdx.x] = r_w[blockIdx.x] % NUM_OF_AGENTS;
	r_w[nh] = r_w[nh] % NUM_OF_AGENTS;

	ind_to_choose[dist_y] = (distance[dist_y] <= distance[ind_to_comp]) ? blockIdx.y : blockIdx.x;


		index = agent + threadIdx.x;
		nh_index = blockIdx.x + r_w[nh] * NUM_OF_AGENTS;
		nh_pos[index] = (ind_to_choose[nh_index] == blockIdx.x) ? //calc X_gwo pos otherwise calc with agent in nh
			agent_pos[index] + r[index] * (agent_new_pos[index] * agent_pos[r_w[blockIdx.x] * NUM_OF_DIMS + threadIdx.x])
			:
			agent_pos[index] + r[index] * (agent_pos[ind_to_choose[nh_index]] * agent_pos[r_w[blockIdx.x] * NUM_OF_DIMS + threadIdx.x]);

	nh_pos[index] = (lo <= nh_pos[index]) ? nh_pos[index] : lo;
	nh_pos[index] = (hi >= nh_pos[index]) ? nh_pos[index] : hi;
}

__global__ void abc_rns(const float* agent_pos, float* agent_new_pos, const int* a, const int* b, const float* r, const unsigned int* rI) {

	float tmp;
	unsigned int index = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	unsigned int r_index = threadIdx.x + rI[blockIdx.x] % NUM_OF_AGENTS * NUM_OF_DIMS;
	r_index = (r_index == index) ? r_index + threadIdx.x : r_index;

	tmp = agent_pos[index] + (2 * r[index] - 1) * (agent_pos[index] - agent_pos[r_index]);

	agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
}

__global__ void abc_rns(const float* agent_pos, float* agent_new_pos,const unsigned int* indices_to_compute, 
	const int* a, const int* b, const float*r, const unsigned int* rI) {
	
	float tmp;
	unsigned int index_to_save = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	unsigned int index_to_compute = threadIdx.x + indices_to_compute[blockIdx.x] * NUM_OF_DIMS;
	unsigned int r_index = threadIdx.x + (rI[blockIdx.x] % NUM_OF_AGENTS) * NUM_OF_DIMS;
	r_index = (r_index == index_to_compute) ? r_index + threadIdx.x : r_index;

	tmp = agent_pos[index_to_save] + (2 * r[index_to_compute] - 1) * (agent_pos[index_to_compute] - agent_pos[r_index]);

	agent_new_pos[index_to_save] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
	agent_new_pos[index_to_save] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
}

__global__ void calc_distances(const float* agent_pos, float* tmp_distance, float* distance)
{
	unsigned int offset_x = blockIdx.x * NUM_OF_DIMS;
	unsigned int index_x = threadIdx.x + blockIdx.x * NUM_OF_DIMS;
	unsigned int offset_y = blockIdx.y * NUM_OF_DIMS;
	unsigned int index_y = threadIdx.x + blockIdx.y * NUM_OF_DIMS;
	
	unsigned int index = threadIdx.x + blockIdx.x * NUM_OF_DIMS + blockIdx.y * NUM_OF_INDICES;	// index of pos to save
	unsigned int step = DIMS_TO_LOG_HALF;

	float R = 0;

#pragma unroll
	for (auto i = 0; i < NUM_OF_DIMS; i++) {
			R += __powf(fabs(agent_pos[offset_x + i] - agent_pos[offset_y + i]), 2);
	}

	distance[blockIdx.x + blockIdx.y * NUM_OF_AGENTS] = sqrt(R);
}

//__global__ void calc_distances(const float* agent_pos, const float* agent_new_pos, float* distance)
//{
//	unsigned int offset_x = blockIdx.x * NUM_OF_DIMS;
//	unsigned int offset_y = blockIdx.y * NUM_OF_DIMS;
//
//	unsigned int index = 0;	// index of pos to save
//
//	float R = 0;
//
//#pragma unroll
//		for (auto i = 0; i < NUM_OF_DIMS; i++) {
//			R+= (blockIdx.x == blockIdx.y) ? 
//				__powf(agent_new_pos[offset_x + i] - agent_pos[offset_y + i], 2) 
//				:
//				__powf(agent_pos[offset_x + i] - agent_pos[offset_y + i], 2);
//		}
//
//	distance[blockIdx.x + blockIdx.y * NUM_OF_AGENTS] = sqrtf(R);
//}

__global__ void compare_two_pop(float* pos, float* val, const float* GWO_pos, const float* GWO_val, 
	const float* nh_pos, const float* nh_val)
{
	unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	val[blockIdx.x] = (GWO_val[blockIdx.x] < nh_val[blockIdx.x]) ? GWO_val[blockIdx.x] : nh_val[blockIdx.x];
	pos[ind] = (GWO_val[blockIdx.x] < nh_val[blockIdx.x]) ? GWO_pos[ind] : nh_pos[ind];
}

__global__ void compare_two_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val)
{
	unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	old_val[blockIdx.x] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_val[blockIdx.x] : old_val[blockIdx.x];
	old_pos[ind] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_pos[ind] : old_pos[ind];
}

__global__ void compare_two_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val, unsigned int* abbadon_dec)
{
	unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	old_val[blockIdx.x] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_val[blockIdx.x] : old_val[blockIdx.x];
	old_pos[ind] = (new_val[blockIdx.x] < old_val[blockIdx.x]) ? new_pos[ind] : old_pos[ind];
	abbadon_dec[blockIdx.x] += (new_val[blockIdx.x] < old_val[blockIdx.x]) ? 0 : 1;
}

__global__ void compare_ff_pop(float* old_pos, float* old_val, const float* new_pos, const float* new_val)
{
	unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int max_agent = NUM_OF_AGENTS * (NUM_OF_AGENTS-1);

#pragma unroll
	for (auto i = blockIdx.x; i < blockIdx.x + max_agent; i += NUM_OF_AGENTS) {
		
		old_val[blockIdx.x] = (new_val[i] < old_val[blockIdx.x]) ? new_val[i] : old_val[blockIdx.x];
		old_pos[ind] = (new_val[i] <= old_val[blockIdx.x]) ? new_pos[i * NUM_OF_DIMS + threadIdx.x] : old_pos[ind];
		
	}
}

__global__ void probability_selection(const float* val, const float* r, unsigned int* index_to_rns) {
	__shared__ float fit[NUM_OF_AGENTS];
	__shared__ float fit_sum[NUM_OF_AGENTS];
	__shared__ float tmp_sum[NUM_OF_AGENTS];
	unsigned int step = 1;
	unsigned int step_index = 0;
	fit_sum[threadIdx.x] = fit[threadIdx.x] = (val[threadIdx.x] < 0) ? 1 + fabs(val[threadIdx.x]) : 1 / (1 + fabs(val[threadIdx.x]));
	__syncthreads();
#pragma unroll
	for (auto i = 0; i < NUM_OF_RUNS; ++i)
	{
	step_index = step + threadIdx.x;
	if (step_index < NUM_OF_AGENTS)
	{
		tmp_sum[step_index] += fit_sum[threadIdx.x];
	}
	__syncthreads();
	if (step_index < NUM_OF_AGENTS)
	{
		fit_sum[step_index] += tmp_sum[step_index];
	}
	step <<= 1;
	}

	__syncthreads();
	fit[threadIdx.x] = fit_sum[threadIdx.x] / fit_sum[NUM_OF_AGENTS-1];

	// choose based on probability
	unsigned int start_index = (unsigned int)floor(r[threadIdx.x] * (NUM_OF_AGENTS - 1));

	int i;
	int j = i = (r[threadIdx.x] >= fit[start_index]) ? 1 : -1;

	switch (j)
	{
	case 1:
		while (r[threadIdx.x] > fit[start_index])
		{
			start_index += i;
		}
		index_to_rns[threadIdx.x] = (NUM_OF_AGENTS <= start_index) ? NUM_OF_AGENTS - 1 : start_index;
		break;
	case -1:
		while (r[threadIdx.x] < fit[start_index])
		{
			start_index += i;
		}
		index_to_rns[threadIdx.x] = start_index + 1;
		break;
	}

}

__global__ void scout_phase(unsigned int* abbadon_dec, const unsigned int abbadon_val, const int* a, const int* b, 
	const float* r, float* agent_pos, unsigned int best_pos) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int best_index = threadIdx.x + best_pos * blockDim.x;

	agent_pos[index] = (abbadon_dec[blockIdx.x] >= abbadon_val && best_pos != blockIdx.x) ? 
		agent_pos[index] + (2 * r[index] - 1) * (agent_pos[index] - agent_pos[best_index])
		//a[threadIdx.x] + r[index] * (b[threadIdx.x] - a[threadIdx.x])
		: 
		agent_pos[index];

	//agent_pos[index] = (a[threadIdx.x] <= agent_pos[index]) ? agent_pos[index] : a[threadIdx.x];
	//agent_pos[index] = (b[threadIdx.x] >= agent_pos[index]) ? agent_pos[index] : b[threadIdx.x];

	abbadon_dec[blockIdx.x] = (abbadon_dec[blockIdx.x] >= abbadon_val) ? 0 : abbadon_dec[blockIdx.x];
}

void error_h(cudaError_t e) {

	if (e != 0)
		exit(e);
}














//__global__ void best_sol(const int NUM_OF_AGENTS, const float* agent_val, unsigned int* indice, float* best_val)
//{
//	int j = blockIdx.x * NUM_OF_AGENTS / blockDim.x;
//	best_val[blockIdx.x] = agent_val[j];
//	indice[blockIdx.x] = j;
//
//#pragma unroll
//	for (unsigned int i = j; i < j + NUM_OF_AGENTS / blockDim.x; ++i)
//	{
//		indice[blockIdx.x] = (agent_val[i] < best_val[blockIdx.x]) ? i : indice[blockIdx.x];
//		best_val[blockIdx.x] = (i == indice[blockIdx.x]) ? agent_val[i] : best_val[blockIdx.x];
//	}
//
//}

	//__global__ void ffa(const float alfa, const float beta, const float gamma, const int* a, const int* b, const unsigned long seed,
	//	const unsigned int* best_sol, float* agent_pos, float* agent_new_pos, const float* agent_val)
	//{
	//	unsigned int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	//	unsigned int index_y = threadIdx.x + blockIdx.y * blockDim.y;
	//	unsigned int agent_start = blockIdx.x * blockDim.x*sizeof(double);
	//
	//	__shared__ float R_i[NUM_OF_INDICES];
	//	__shared__ float  R[NUM_OF_AGENTS];
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
	//	tmp = agent_pos[index] + beta * __expf(-gamma * R[blockIdx.x]) * agent_diff + alfa * curand_normal(&r);
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
//	unsigned int offset_x = agent_x * NUM_OF_DIMS;
//	unsigned int offset_y = agent_y * NUM_OF_DIMS;
//
//	unsigned int offset_Y = agent_y * (NUM_OF_AGENTS)*NUM_OF_DIMS;
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
//		for (auto i = 0; i < NUM_OF_DIMS; i++) {
//			R += pow(agent_pos[offset_x + i] - agent_pos[offset_y + i], 2);	//calc distance
//		}
//
//#pragma unroll
//		for (auto i = 0; i < NUM_OF_DIMS; i++) {
//
//			x_indice = offset_x + i;
//			tmp = agent_pos[x_indice] + beta * __expf(-gamma * R) * (agent_pos[x_indice] - agent_pos[offset_y + i])
//				+ alfa * curand_normal(&r);		//new pos
//
//			index = x_indice + offset_Y;	//possible pos, 1 column for agent
//			agent_new_pos[index] = (a[threadIdx.x] <= tmp) ? tmp : a[threadIdx.x];
//			agent_new_pos[index] = (b[threadIdx.x] >= tmp) ? tmp : b[threadIdx.x];
//		}
//
//	}
//	else {
//		for (auto i = 0; i < NUM_OF_DIMS; i++) {
//			x_indice = offset_x + i;
//			index = x_indice + offset_Y;
//			agent_new_pos[index] = agent_pos[x_indice];		//save old 
//		}
//
//	}
//	__syncthreads();
//}
