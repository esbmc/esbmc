#ifndef _CUDA_RUNTIME_API_H
#define _CUDA_RUNTIME_API_H 1

#include "driver_types.h"
#include "host_defines.h"
#include "builtin_types.h"
#include "cuda_device_runtime_api.h"
#include "sm_atomic_functions.h"
#include "cuda_error.h"
#include "call_kernel.h"

#include <stddef.h>
#include <stdio.h>
#include <cstdlib>

/** \cond impl_private */
#if !defined(__dv)

#  if defined(__cplusplus)

#    define __dv(v)

#  else /* __cplusplus */

#    define __dv(v)

#  endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

cudaError_t
__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  __ESBMC_atomic_begin();
  __ESBMC_assert(count > 0, "Size to be allocated may not be less than zero");

  char *cdst = (char *)dst;
  const char *csrc = (const char *)src;
  int numbytes = count / (sizeof(char));

  for (int i = 0; i < numbytes; i++)
    cdst[i] = csrc[i];

  lastError = CUDA_SUCCESS;
  __ESBMC_atomic_end();
  return CUDA_SUCCESS;
}

template <class T1, class T2>
cudaError_t cudaMemcpy(T1 dst, T2 src, size_t count, enum cudaMemcpyKind kind)
{
  return __cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
  __ESBMC_atomic_begin();
  cudaError_t tmp;
  //pre-conditions
  __ESBMC_assert(size > 0, "Size to be allocated may not be less than zero");
  *devPtr = malloc(size);

  if (*devPtr == NULL)
    tmp = CUDA_ERROR_OUT_OF_MEMORY;
  else
    tmp = CUDA_SUCCESS;

  //post-conditions
  __ESBMC_assert(tmp == CUDA_SUCCESS, "Memory was not allocated");

  lastError = tmp;
  __ESBMC_atomic_end();
  return tmp;
}

cudaError_t cudaFree(void *devPtr)
{
  __ESBMC_atomic_begin();
  free(devPtr);
  lastError = CUDA_SUCCESS;
  __ESBMC_atomic_end();
  return CUDA_SUCCESS;
}

const char *cudaGetErrorString(cudaError_t error)
{
  char *erroReturn;

  switch (error)
  {
  case 0:
    return "CUDA_SUCCESS";
    break;
  case 1:
    return "CUDA_ERROR_INVALID_VALUE";
    break;
  case 2:
    return "CUDA_ERROR_OUT_OF_MEMORY";
    break;
  case 3:
    return "CUDA_ERROR_NOT_INITIALIZED";
    break;
  case 4:
    return "CUDA_ERROR_DEINITIALIZED";
    break;
  case 5:
    return "CUDA_ERROR_PROFILER_DISABLED";
    break;
  case 6:
    return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    break;
  case 7:
    return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    break;
  case 8:
    return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    break;
  case 100:
    return "CUDA_ERROR_NO_DEVICE";
    break;
  case 101:
    return "CUDA_ERROR_INVALID_DEVICE";
    break;
  case 200:
    return "CUDA_ERROR_INVALID_IMAGE";
    break;
  case 201:
    return "CUDA_ERROR_INVALID_CONTEXT";
    break;
  case 202:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    break;
  case 205:
    return "CUDA_ERROR_MAP_FAILED";
    break;
  case 206:
    return "CUDA_ERROR_UNMAP_FAILED";
    break;
  case 207:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";
    break;
  case 208:
    return "CUDA_ERROR_ALREADY_MAPPED";
    break;
  case 209:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    break;
  case 210:
    return "CUDA_ERROR_ALREADY_ACQUIRED";
    break;
  case 211:
    return "CUDA_ERROR_NOT_MAPPED";
    break;
  case 212:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    break;
  case 213:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    break;
  case 214:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";
    break;
  case 215:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    break;
  case 216:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    break;
  case 217:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    break;
  case 218:
    return "CUDA_ERROR_INVALID_PTX";
    break;
  case 219:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    break;
  case 300:
    return "CUDA_ERROR_INVALID_SOURCE";
    break;
  case 301:
    return "CUDA_ERROR_FILE_NOT_FOUND";
    break;
  case 302:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    break;
  case 303:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    break;
  case 304:
    return "CUDA_ERROR_OPERATING_SYSTEM";
    break;
  case 400:
    return "CUDA_ERROR_INVALID_HANDLE";
    break;
  case 500:
    return "CUDA_ERROR_NOT_FOUND";
    break;
  case 600:
    return "CUDA_ERROR_NOT_READY";
    break;
  case 700:
    return "CUDA_ERROR_ILLEGAL_ADDRESS";
    break;
  case 701:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    break;
  case 702:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";
    break;
  case 703:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    break;
  case 704:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    break;
  case 705:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    break;
  case 708:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    break;
  case 709:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    break;
  case 710:
    return "CUDA_ERROR_ASSERT";
    break;
  case 711:
    return "CUDA_ERROR_TOO_MANY_PEERS";
    break;
  case 712:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    break;
  case 713:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    break;
  case 714:
    return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    break;
  case 715:
    return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    break;
  case 716:
    return "CUDA_ERROR_MISALIGNED_ADDRESS";
    break;
  case 717:
    return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    break;
  case 718:
    return "CUDA_ERROR_INVALID_PC";
    break;
  case 719:
    return "CUDA_ERROR_LAUNCH_FAILED";
    break;
  case 800:
    return "CUDA_ERROR_NOT_PERMITTED";
    break;
  case 801:
    return "CUDA_ERROR_NOT_SUPPORTED";
    break;
  case 999:
    return "CUDA_ERROR_UNKNOWN";
    break;
  default:
    return "CUDA_SUCCESS";
  }
}

////////////////////////////////////////////////////////////////////////////
//!  Structure that represents the devices of CUDA.

//Struct to Device - OK
typedef struct cudaDevicesList
{
  int id;
  int active;
  struct cudaDeviceProp deviceProp;
  struct cudaDevicesList *prox;
} cudaDeviceList_t;

cudaDeviceList_t *cudaDeviceList = NULL;

//Insert a device
void cudaDeviceInsert(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  //Verifies that the device exists in the list
  while (auxDevice != NULL)
  {
    if (auxDevice->id == device)
    {
      //printf("\nDevice existing");
      //return 0;
    }
    auxDevice = auxDevice->prox;
  }
  //Insert new device
  cudaDeviceList_t *newCudaDevice;

  newCudaDevice = (cudaDeviceList_t *)__ESBMC_alloca(sizeof(cudaDeviceList_t));
  if (newCudaDevice == NULL)
    return;

  newCudaDevice->id = device;
  newCudaDevice->active = 0;
  //newCudaDevice->deviceProp.regsPerBlock = var; //Insert fields to deviceProp
  newCudaDevice->prox = NULL;

  if (cudaDeviceList == NULL)
  {
    cudaDeviceList = newCudaDevice;
  }
  else
  {
    newCudaDevice->prox = cudaDeviceList;
    cudaDeviceList = newCudaDevice;
  }
  //	return 1;
}

//Searching for a device in the devices list
void cudaPrintDevice()
{
  //printf("\n\n*** CUDA Device\n");
  __ESBMC_atomic_begin();
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  {
    //printf("->Device: %d Active:%d\n",auxDevice->id,auxDevice->active);
    auxDevice = auxDevice->prox;
  }
  __ESBMC_atomic_end();
}

//Searching for a device in the devices list
int searchCudaDevice(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  {
    if (auxDevice->id == device)
    {
      return 1;
    }
    else
    {
      auxDevice = auxDevice->prox;
    }
  }
  return 0;
}

//Checks whether the device is in use
int cudaDeviceActive(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  {
    if (auxDevice->id == device)
    {
      if (auxDevice->active == 1)
      {
        return 1;
      }
      else
      {
        return 0;
      }
    }
  }
  return 0;
}

//Start a device
int cudaDeviceStart(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  {
    if (auxDevice->id == device)
    {
      auxDevice->active = 1;
      return 1;
    }
    auxDevice = auxDevice->prox;
  }
  return 0;
}

// Choose a device to work
cudaError_t cudaSetDevice(int device)
{
  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  { //Scroll through the list
    if (auxDevice->id == device)
    { //Checks if the device
      if (auxDevice->active == 1)
      {                                     //Verifies that the device is active
        return cudaErrorDeviceAlreadyInUse; //cudaErrorDeviceAlreadyInUse
        lastError = cudaErrorDeviceAlreadyInUse;
      }
      auxDevice->active = 1;
      lastError = cudaSuccess;
      return cudaSuccess;
    }
    else
      auxDevice = auxDevice->prox;
  }
  //If not found, return cudaErrorInvalidDevice
  lastError = cudaErrorInvalidDevice;
  return cudaErrorInvalidDevice;
}

// Returns the number of compute-capable devices.
cudaError_t cudaGetDeviceCount(int *count)
{
  /*
	cudaDeviceList_t *auxDevice = cudaDeviceList;
	int i;

	while(auxDevice!=NULL){
		i++;
		auxDevice = auxDevice->prox;
	}
	 */
  lastError = cudaSuccess;
  return cudaSuccess;
}

//Destroy all allocations and reset all state on the current device in the current process.
cudaError_t cudaDeviceReset()
{
  /*
	int tmp;
	threadsList_t *node;

	while(cudaThreadList != NULL){
		//		pthread_exit(cudaThreadList->thread);
		node = cudaThreadList;
		cudaThreadList = cudaThreadList->prox;
		free(node);
	}

	lastError = cudaSuccess;
	*/
  return cudaSuccess;
}

/*
 *    Memory Management
 */
cudaError_t cudaMemcpyToSymbol(
  const void *symbol,
  const void *src,
  size_t count,
  size_t offset __dv(0),
  enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
  __ESBMC_atomic_begin();
  cudaError_t out;

  out = __cudaMemcpy((void *)(symbol), (const void *)src, count, kind);

  lastError = out;
  __ESBMC_atomic_end();
  return out;
}

cudaError_t cudaMemcpyFromSymbol(
  void *dst,
  const void *symbol,
  size_t count,
  size_t offset __dv(0),
  enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
  __ESBMC_atomic_begin();
  cudaError_t out;

  out = __cudaMemcpy((void *)dst, (const void *)(symbol), count, kind);

  lastError = out;
  __ESBMC_atomic_end();
  return out;
}

extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

extern cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
  struct cudaDeviceProp deviceChosen;

  int out;

  cudaDeviceList_t *auxDevice = cudaDeviceList;

  while (auxDevice != NULL)
  {
    if (auxDevice->id == device)
    {
      deviceChosen = auxDevice->deviceProp;
      prop = &deviceChosen;
      return cudaSuccess;
    }
    auxDevice = auxDevice->prox;
  }
  return CUDA_ERROR_INVALID_VALUE;
}

cudaError_t cudaGetLastError()
{
  return lastError;
}

typedef struct threadsList
{
  pthread_t thread;
  struct threadsList *prox;
} threadsList_t;

threadsList_t *cudaThreadList = NULL;

cudaError_t cudaThreadSynchronize()
{
  __ESBMC_atomic_begin();
  cudaError_t tmp;

  while (cudaThreadList != NULL)
  {
    threadsList_t *node;
    pthread_join(cudaThreadList->thread, NULL);
    node = cudaThreadList;
    cudaThreadList = cudaThreadList->prox;
    free(node);
  }
  lastError = CUDA_SUCCESS;
  __ESBMC_atomic_end();
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

void __syncthreads()
{
}

void __threadfence()
{
}

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceSynchronize(void);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

extern __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr);

extern __host__ cudaError_t CUDARTAPI cudaThreadExit(void);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaPeekAtLastError(void);

extern __host__ __cudart_builtin__ const char *CUDARTAPI
cudaGetErrorName(cudaError_t error);

extern __host__ __cudart_builtin__ const char *CUDARTAPI
cudaGetErrorString(cudaError_t error);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);

extern __host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaGetDevice(int *device);

extern __host__ cudaError_t CUDARTAPI
cudaSetValidDevices(int *device_arr, int len);

extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);

extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaEventDestroy(cudaEvent_t event);

extern __host__ cudaError_t CUDARTAPI
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

extern __host__ cudaError_t CUDARTAPI
cudaSetupArgument(const void *arg, size_t size, size_t offset);

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d);

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size);

extern __host__ cudaError_t CUDARTAPI
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);

extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaHostAlloc(void **pHost, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaHostRegister(void *ptr, size_t size, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaHostGetFlags(unsigned int *pFlags, void *pHost);

extern __host__ cudaError_t CUDARTAPI
cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent);

extern __host__ cudaError_t CUDARTAPI
cudaMemcpy3D(const struct cudaMemcpy3DParms *p);

extern __host__ cudaError_t CUDARTAPI
cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);

extern __host__ cudaError_t CUDARTAPI
cudaMemGetInfo(size_t *free, size_t *total);

extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(
  void *dst,
  int dstDevice,
  const void *src,
  int srcDevice,
  size_t count);

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(
  void *dst,
  size_t dpitch,
  const void *src,
  size_t spitch,
  size_t width,
  size_t height,
  enum cudaMemcpyKind kind);

extern __host__ cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count);

extern __host__ cudaError_t CUDARTAPI cudaMemset2D(
  void *devPtr,
  size_t pitch,
  int value,
  size_t width,
  size_t height);

extern __host__ cudaError_t CUDARTAPI cudaMemset3D(
  struct cudaPitchedPtr pitchedDevPtr,
  int value,
  struct cudaExtent extent);

extern __host__ cudaError_t CUDARTAPI
cudaGetSymbolAddress(void **devPtr, const void *symbol);

extern __host__ cudaError_t CUDARTAPI
cudaGetSymbolSize(size_t *size, const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(
  struct cudaPointerAttributes *attributes,
  const void *ptr);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

extern __host__ cudaError_t CUDARTAPI
cudaDeviceDisablePeerAccess(int peerDevice);

extern __host__ cudaError_t CUDARTAPI cudaBindTexture(
  size_t *offset,
  const struct textureReference *texref,
  const void *devPtr,
  const struct cudaChannelFormatDesc *desc,
  size_t size __dv(UINT_MAX));

extern __host__ cudaError_t CUDARTAPI cudaBindTexture2D(
  size_t *offset,
  const struct textureReference *texref,
  const void *devPtr,
  const struct cudaChannelFormatDesc *desc,
  size_t width,
  size_t height,
  size_t pitch);

extern __host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference *texref);

extern __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
  size_t *offset,
  const struct textureReference *texref);

extern __host__ cudaError_t CUDARTAPI cudaGetTextureReference(
  const struct textureReference **texref,
  const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(
  const struct surfaceReference **surfref,
  const void *symbol);

extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion);

extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaRuntimeGetVersion(int *runtimeVersion);

#undef __dv

#endif /* cuda_runtime_api.h */
