
#if !defined(__SM_12_ATOMIC_FUNCTIONS_H__)
#define __SM_12_ATOMIC_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "host_defines.h"

#define __location__(device) void
#define __device__ __location__(device)

/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int
__ullAtomicAdd(unsigned long long int *address, unsigned long long int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int
__ullAtomicExch(unsigned long long int *address, unsigned long long int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned long long int __ullAtomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val);

/*DEVICE_BUILTIN*/
extern __device__ int __any(int cond);
/*DEVICE_BUILTIN*/
extern __device__ int __all(int cond);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ unsigned long long int
__ullAtomicAdd(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  return old_value;
}

extern __device__ unsigned long long int
__ullAtomicExch(unsigned long long int *address, unsigned long long int val)
{
  unsigned long long int old_value;

  old_value = *address;
  *address = val;

  return old_value;
}

extern __device__ unsigned long long int __ullAtomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val)
{
  unsigned long long int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  return old_value;
}

/*
extern __device__ int                    __any(int cond){
}

extern __device__ int                    __all(int cond){
}
*/

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __device__ unsigned long long int
atomicAdd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAdd(address, val);
}

static __inline__ __device__ unsigned long long int
atomicExch(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicExch(address, val);
}

static __inline__ __device__ unsigned long long int atomicCAS(
  unsigned long long int *address,
  unsigned long long int compare,
  unsigned long long int val)
{
  return __ullAtomicCAS(address, compare, val);
}

static __inline__ __device__ bool any(bool cond)
{
  return (bool)__any((int)cond);
}

static __inline__ __device__ bool all(bool cond)
{
  return (bool)__all((int)cond);
}

#endif /* !__SM_12_ATOMIC_FUNCTIONS_H__ */
