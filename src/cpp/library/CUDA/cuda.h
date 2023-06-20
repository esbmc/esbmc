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

//Structure that represents the threads of CUDA.

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

#endif /* cuda.h */
