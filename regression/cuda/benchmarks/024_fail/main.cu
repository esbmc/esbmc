
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <assert.h>

__global__ void Asum(int *a, int *b, int *c){
	*c = *a + *b;
}

int main(void){
	int a, b, c;
	int *dev_a, *dev_b, *dev_c;		//These are pointers to a memory slot ON DEVICE
	int size = sizeof(int);			//memory size in bytes

	cudaMalloc((void**)&dev_a,size);	//cudaMalloc() allocates a memory slot on device (GPU memory)
						//this slot equals size bytes
						//void** assures that pointers won't have trouble getting a variable that is not an int type
						//dev_a now points to the allocated slot
	cudaMalloc((void**)&dev_b,size);
	cudaMalloc((void**)&dev_c,size);	//conclusion: pointers are referencing a position that is avaliable from DEVICE
						//a, b and c positions are not avaliable from device, a priori
	a = 2;
	b = 7;
	c = 8;

	cudaMemcpy(dev_a,&a,size, cudaMemcpyHostToDevice);	//note that &a is used
	cudaMemcpy(dev_b,&b,size, cudaMemcpyHostToDevice);	//cudaMemcpy(*destiny, *source, size, cudaMemcpyKind)

	//	Asum<<<1,1>>>(dev_a,dev_b,dev_c);
	ESBMC_verify_kernel(Asum, 1,2,dev_a,dev_b, dev_c);

	cudaMemcpy(&c,dev_c,size,cudaMemcpyDeviceToHost);

	printf("a + b = %d\n", c);

	assert(c != a+b);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
