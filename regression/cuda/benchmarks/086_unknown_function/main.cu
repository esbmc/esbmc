//pass
//blockDim=1024 --gridDim=1 --no-inline

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cmath>

typedef double(*funcType)(double);

__device__ double bar(double x) {
  return sin(x);
}

__device__ funcType select_func(int i) {
  if (i == 0)
    return bar;
}

__global__ void foo(double x, int i)
{
	funcType f = select_func(i);
}

int main(){

	int select_function = 1; // 1= sen; 0=cos
	double angle = 1.57; //0;

	//foo <<<1,2>>>(angle, select_function);
	ESBMC_verify_kernel_c (foo,1,2,angle, select_function);

	cudaThreadSynchronize();

	return 0;
}
