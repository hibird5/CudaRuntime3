//#include"h_fce.cuh"
//
//
//__global__ void get_constr(const int min, const int max, int* a, int* b)
//{
//	if (!a || !b)
//		return;
//
//	a[blockIdx.x] = min;
//	b[blockIdx.x] = max;
//	return;
//}
//
//__global__ void init_pop_pos(float* agent_pos, const int num_of_indices, const int* a, const int* b,const float* r)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	agent_pos[index] = (index < num_of_indices) ?
//		a[threadIdx.x] + r[index] * (b[threadIdx.x] - a[threadIdx.x]) : 0;
//
//}
//
//__global__ void cost_func(const int num_of_dims, const float* agent_pos, const int input_func, float* agent_val)
//{
//	int agent = blockIdx.x * num_of_dims;
//
//	agent_val[blockIdx.x] = 0;
//
//	switch (input_func)
//	{
//	case 1:
//#pragma unroll 
//		for (int i = 0; i < num_of_dims; ++i)
//		{
//			agent_val[blockIdx.x] += agent_pos[agent + i] * agent_pos[agent + i] * agent_pos[agent + i] * agent_pos[agent + i]
//				- 16 * (agent_pos[agent + i] * agent_pos[agent + i])
//				+ 5 * agent_pos[agent + i];
//		}
//		agent_val[blockIdx.x] /= 2;
//		break;
//	case 2:
//#pragma unroll 
//		for (int i = 0; i < num_of_dims; ++i)
//		{
//			agent_val[blockIdx.x] += 220 / (i + blockIdx.x + 1);
//		}
//		agent_val[blockIdx.x] /= 2;
//		break;
//
//	default:
//#pragma unroll 
//		for (int i = 0; i < num_of_dims; ++i)
//		{
//			agent_val[blockIdx.x] += agent_pos[agent + i] * agent_pos[agent + i];
//		}
//		break;
//	};
//}
//
////__global__ void init_pop_vals(population* pop)
////{
////	cost_func(pop->agent_pos, pop->agent_val[blockIdx.x], pop->sizes.dimensions, pop->input_func);
////}