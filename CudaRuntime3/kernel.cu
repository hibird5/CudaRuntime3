//#include <cstdlib>
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//
//
//#ifndef __CUDACC__ 
//#define __CUDACC__
//#endif
//
//#include <cuda.h>
//#include <curand.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "h_fce.cuh"
//#include <device_functions.h>
//#include <cuda_runtime_api.h>
////#include <cub/device/device_radix_sort.cuh>
////#include <cub/cub.cuh> 
//
//#include "h_fce.cuh"
//
//using namespace std;
//
//
//
//
//int main()
//{
//	void* d_temp = NULL;
//	size_t temp_storage_bites = 0;
//	/*cub::KeyValuePair<int, int>* index;
//	cudaMalloc(&index, sizeof(cub::KeyValuePair<int, int>));*/
//
//	get_constr << <dim, 1 >> > (-10, 10, pop.lim);
//	init_pop_pos << <ag, dim >> > (pop, r);
//	cost_func << <ag, 1 >> > (pop);
//	//searchForBestKernel <<<2, 10 >>> (pop.agent_val, ind);
//
//
//	unsigned int* in = NULL;
//	in = (unsigned int*)malloc(sizeof(unsigned int));
//	cudaMemcpy(pop_back, pop.agent_pos, pop.sizes.dimensions * pop.sizes.individuals * sizeof(float), ::cudaMemcpyDeviceToHost);
//	cudaMemcpy(pop_vals, pop.agent_val, pop.sizes.individuals * sizeof(float), ::cudaMemcpyDeviceToHost);
//	cudaMemcpy(in, ind, pop.sizes.individuals * sizeof(unsigned int), ::cudaMemcpyDeviceToHost);
//	//cudaMemcpy(in, index, sizeof(int), ::cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < ag; ++i)
//	{
//		for (int j = 0; j < dim; ++j)
//		{
//			cout << pop_back[i * dim + j] << ", ";
//		}
//		cout << '\n' << endl;
//	}
//
//	for (int i = 0; i < ag; ++i)
//	{
//		cout << pop_vals[i] << ", ";
//	}
//	cout << '\n' << endl;
//
//	for (int i = 0; i < ag; ++i)
//	{
//		cout << in[i] << ", ";
//		cout << endl;
//	}
//
//
//
//	cudaFree(r);
//	cudaFree(pop.agent_pos);
//	cudaFree(pop.agent_val);
//	cudaFree(pop.lim.a);
//	cudaFree(pop.lim.b);
//
//	free(pop_back);
//	free(pop_vals);
//
//	return 0;
//
//}