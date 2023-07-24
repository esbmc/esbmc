//xfail:BOOGIE_ERROR
//--blockDim=128 --gridDim=16 --no-inline
//assert\(false\)

#include <stdio.h>
#include <assert.h>
#include "cuda_runtime_api.h"

typedef void(*funcType)(float*);

__device__ void a(float *v)
{
	printf ("funcA with p%f = %f", *v, *v);
}
__device__ void b(float *v)
{
	printf ("funcB with p%f = %f", *v, *v);
}

__device__ void c(float *v)
{
	printf ("funcC with p%f = %f", *v, *v);
}

__device__ void d(float *v)
{
	printf ("funcD with p%f = %f", *v, *v);
}

__device__ void e(float *v)
{
	printf ("funcE with p%f = %f", *v, *v);
}

__global__ void should_fail(float * __restrict p1, float * __restrict p2, float * __restrict p3, float * __restrict p4, float * __restrict p5, int x, int y)
{
	funcType fp = a;

    switch(x) {
    case 1:
        fp = &a;
        break;
    case 2:
        fp = &b;
        break;
    case 3:
        fp = &c;
        break;
    case 4:
        fp = &d;
        break;
    default:
        fp = &e;
        break;
    }

    switch(y) {
    case 1:
        fp(p1);
        break;
    case 2:
        fp(p2);
        break;
    case 3:
        fp(p3);
        break;
    case 4:
        fp(p4);
        break;
    default:
        fp(p5);
        break;
    }

   assert(0);
}

int main (){

	float p1, p2, p3, p4, p5;
	float *dev_p1, *dev_p2, *dev_p3, *dev_p4, *dev_p5;

	p1 = 1; p2 = 2; p3 = 3; p4 = 4; p5 = 5;

	cudaMalloc((void**)&dev_p1, sizeof(float));
	cudaMalloc((void**)&dev_p2, sizeof(float));
	cudaMalloc((void**)&dev_p3, sizeof(float));
	cudaMalloc((void**)&dev_p4, sizeof(float));
	cudaMalloc((void**)&dev_p5, sizeof(float));

	cudaMemcpy(dev_p1,&p1, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p2,&p2, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p3,&p3, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p4,&p4, sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p5,&p5, sizeof(float),cudaMemcpyHostToDevice);

	//should_fail <<<1,2>>>(dev_p1, dev_p2, dev_p3, dev_p4, dev_p5, 4, 4);
	ESBMC_verify_kernel_f(should_fail,1,2,dev_p1, dev_p2, dev_p3, dev_p4, dev_p5, 4, 4);
	
	cudaMemcpy(&p1,dev_p1,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&p2,dev_p2,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&p3,dev_p3,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&p4,dev_p4,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&p5,dev_p5,sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(dev_p1);
	cudaFree(dev_p2);
	cudaFree(dev_p3);
	cudaFree(dev_p4);
	cudaFree(dev_p5);

	return 0;
}
