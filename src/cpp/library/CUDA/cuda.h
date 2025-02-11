#ifndef _CUDA_H
#define _CUDA_H 1

#include "cuda_error.h"
#include "sm_atomic_functions.h"
#include "curand_kernel.h"
#include "curand.h"
#include "call_kernel.h"
#include "vector_types.h"
#include "device_launch_parameters.h"

#include <stddef.h>
#include <cstdlib>
#include <string.h>
#include <pthread.h>
#include <assert.h>
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
  newCudaThread = (threadsList_t *)__ESBMC_alloca(sizeof(threadsList_t));
  if (newCudaThread == NULL)
    return;
  newCudaThread->thread = threadAux;
  newCudaThread->prox = NULL;

  if (cudaThreadList == NULL)
  {
    cudaThreadList = newCudaThread;
  }
  else
  {
    newCudaThread->prox = cudaThreadList;
    cudaThreadList = newCudaThread;
  }
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond;
int count = 0;

#endif /* cuda.h */
