#ifndef _CUDA_H
#define _CUDA_H 1

#include <sm_atomic_functions.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <call_kernel.h>

#include <stddef.h>
#include <cstdlib>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include "vector_types.h"
#include "device_launch_parameters.h"
#include <new>

////////////////////////////////////////////////////////////////////////////
//! Structure that represents the threads of CUDA.

typedef struct threadsList
{
  pthread_t thread;
  struct threadsList *prox;
} threadsList_t;

threadsList_t *cudaThreadList = NULL;

void cudaInsertThread(pthread_t threadAux)
{
  threadsList_t *newCudaThread;
  newCudaThread = (threadsList_t *)malloc(sizeof(threadsList_t));
  if(newCudaThread == NULL)
    exit(0);
  newCudaThread->thread = threadAux;
  newCudaThread->prox = NULL;

  if(cudaThreadList == NULL)
  {
    cudaThreadList = newCudaThread;
  }
  else
  {
    newCudaThread->prox = cudaThreadList;
    cudaThreadList = newCudaThread;
  }
}

cudaError_t cudaThreadSynchronize()
{
  cudaError_t tmp;

  //	if (cudaThreadList == NULL)
  //		return CUDA_ERROR_OUT_OF_MEMORY;

  while(cudaThreadList != NULL)
  {
    threadsList_t *node;
    pthread_join(cudaThreadList->thread, NULL);
    node = cudaThreadList;
    cudaThreadList = cudaThreadList->prox;
    free(node);
  }
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond;
int count = 0;

void __syncthreads()
{
  /*
	int i;
	int aux;

	if(count==0){
		pthread_mutex_init(&mutex,0);
		pthread_cond_init(&cond,0);
	}

	count ++;
	aux = count;

	pthread_mutex_lock(&mutex);

	if(count != GPU_threads){
		pthread_cond_wait(&cond,&mutex);
	} else {
		for(i=0; i<GPU_threads; i++){
			pthread_cond_signal(&cond);
		}
	}

	pthread_mutex_unlock(&mutex);*/
}

void __threadfence()
{
}

void *Address[10];
int Counter = 0;

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
  cudaError_t tmp;
  //pre-conditions
  __ESBMC_assert(size > 0, "Size to be allocated may not be less than zero");
  *devPtr = malloc(size);

  Address[Counter] = *devPtr;
  Counter++;

  if(*devPtr == NULL)
  {
    tmp = CUDA_ERROR_OUT_OF_MEMORY;
    exit(1);
  }
  else
  {
    tmp = CUDA_SUCCESS;
  }

  //post-conditions
  __ESBMC_assert(tmp == CUDA_SUCCESS, "Memory was not allocated");

  lastError = tmp;
  return tmp;
}

/*void verify_kernel_void(void *(*kernel)(void *), int blocks, int threads, void* arg)
 {
 unsigned int n_threads = blocks*threads;
 pthread_t thread[n_threads];
 int i = 0, tmp;

 for(i = 0;i < n_threads;i++){
 pthread_create(&thread[i], NULL, kernel, arg);
 }
 for(i = 0;i < n_threads;i++)
 pthread_join(thread[i], NULL);
 }*/

/* verify_kernel() */

#if 0

void verify_kernel_with_one_arg_void(void *(*kernel)(void *), int blocks, int threads, void* arg)
{
	unsigned int n_threads = blocks*threads;
	pthread_t thread[n_threads];
	int i = 0;
	unsigned int tmp;
	__ESBMC_assume(tmp < n_threads);
	while (i < tmp)
	{
		pthread_create(&thread[i], NULL, kernel, arg);
		cudaInsertThread(thread[i]);
		i++;
	}
}
void verify_kernel_with_two_args_void(void *(*kernel)(void *), int blocks, int threads, void* arg1, void* arg2)
{
	int n_threads;
	n_threads = blocks*threads;
	pthread_t thread[n_threads];
	int i;
	for(i = 0; i < n_threads;i++) {
		pthread_create_with_two_args(&thread[i], NULL, kernel, arg1, arg2);
		cudaInsertThread(thread[i]);
	}
}
void verify_kernel_with_three_args_void(void *(*kernel)(void *), int blocks, int threads, void* arg1, void* arg2, void* arg3)
{
	int n_threads;
	n_threads = blocks*threads;
	pthread_t thread[n_threads];
	int i;
	for(i = 0; i < n_threads;i++) {
		pthread_create_with_three_arg(&thread[i], NULL, kernel, arg1, arg2, arg3);
		cudaInsertThread(thread[i]);
	}
}

template<class RET,class BLOCK, class THREAD, class T1>
void verify_kernel_with_one_arg(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
	gridDim = dim3( blocks );
	blockDim = dim3( threads);

	verify_kernel_with_one_arg_void(
			(voidFunction_t)kernel,
			gridDim.x*gridDim.y*gridDim.z,
			blockDim.x*blockDim.y*blockDim.z,
			(void*)arg);
}
template<class RET,class BLOCK, class THREAD, class T1, class T2>
void verify_kernel_with_two_args(RET *kernel, BLOCK blocks, THREAD threads, T1 arg1, T2 arg2)
{
	gridDim = dim3( blocks );
	blockDim = dim3( threads);

	verify_kernel_with_two_args_void(
			(voidFunction_t)kernel,
			gridDim.x*gridDim.y*gridDim.z,
			blockDim.x*blockDim.y*blockDim.z,
			(void*)arg1,
			(void*)arg2);
}

template<class RET,class BLOCK, class THREAD, class T1, class T2, class T3>
void verify_kernel_with_three_args(RET *kernel, BLOCK blocks, THREAD threads, T1 arg1, T2 arg2, T3 arg3)
{
	gridDim = dim3( blocks );
	blockDim = dim3( threads);

	verify_kernel_with_three_args_void(
			(voidFunction_t)kernel,
			gridDim.x*gridDim.y*gridDim.z,
			blockDim.x*blockDim.y*blockDim.z,
			(void*)arg1,
			(void*)arg2,
			(void*)arg3);
}

#endif

//threadsList_t *cudaThreadList = NULL;

cudaError_t cudaFree(void *devPtr)
{
  free(devPtr);
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

cudaError_t cudaDeviceGetLimit(int device)
{
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

cudaError_t cudaDeviceGetCacheConfig(int device)
{
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

cudaError_t cudaDeviceGetSharedMemConfig(int device)
{
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

cudaError_t cudaPeekAtLastError(int device)
{
  lastError = CUDA_SUCCESS;
  return CUDA_SUCCESS;
}

//-----------------------------------------------
/*
pthread_mutex_t mutex[2] = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond[2];
int count[2];

void __syncthreads() {

	int idThread;
	int condVerifyThread = 1;
	int i = 0;

	pthread_t threadAux = pthread_self();

	while( (i<4) && condVerifyThread){
		if(threadAux==threads_id[i]){
			condVerifyThread = 0;
			idThread = i;
		}
		i++;
	}

	unsigned int id_linear_block = (unsigned int) idThread  / (blockDim.x*blockDim.y*blockDim.z);

	if(count[id_linear_block]==0){
		pthread_mutex_init(&(mutex[id_linear_block]),0);
		pthread_cond_init(&(cond[id_linear_block]),0);
	}

	count[id_linear_block] ++;

	pthread_mutex_lock(&mutex[id_linear_block]);

	if(count[id_linear_block] != 2){
		pthread_cond_wait(&cond[id_linear_block],&mutex[id_linear_block]);
	} else {
		for(i=0; i<2; i++){
			pthread_cond_signal(&cond[id_linear_block]);
		}
	}
	pthread_mutex_unlock(&mutex[id_linear_block]);
}
*/
#endif /* cuda.h */
