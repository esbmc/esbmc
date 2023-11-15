#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>

//typedef int type;
typedef float type;

void generic_function (type *var) {
	assert (*var == 5);
}
/*
int main() {
	type no_pointer = 5;
	type *pointer = (type*)malloc(sizeof(type));
	
	*pointer = 0;

	memcpy(pointer, &no_pointer,sizeof(type));

	assert(*pointer == 5);

	generic_function (pointer);
}*/

int main() {
	float no_pointer = 5;
	float *pointer = (float*)malloc(sizeof(float));
	
	*pointer = 0;

//	cudaMemcpy(pointer, &no_pointer,sizeof(float), cudaMemcpyHostToDevice);
	
	memcpy(pointer, &no_pointer,sizeof(float));

	assert(*pointer == 5);

	generic_function (pointer);
}


