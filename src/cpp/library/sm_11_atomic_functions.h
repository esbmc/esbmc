#if !defined(__SM_11_ATOMIC_FUNCTIONS_H__)
#define __SM_11_ATOMIC_FUNCTIONS_H__

#include "builtin_types.h"
#include "host_defines.h"

#define __location__(device) void
#define __device__ __location__(device)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicAdd(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicAdd(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicExch(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicExch(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ float __fAtomicExch(float *address, float val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicMin(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicMin(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicMax(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicMax(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicInc(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicDec(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicAnd(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicAnd(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicOr(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicOr(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicXor(int *address, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicXor(unsigned int *address, unsigned int val);
/*DEVICE_BUILTIN*/
extern __device__ int __iAtomicCAS(int *address, int compare, int val);
/*DEVICE_BUILTIN*/
extern __device__ unsigned int
__uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern __device__ int __iAtomicAdd(int *address, int val)
{
  int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  return old_value;
}

extern __device__ unsigned int
__uAtomicAdd(unsigned int *address, unsigned int val)
{
  unsigned int old_value, new_value;

  old_value = *address;
  new_value = old_value + val;
  *address = new_value;

  return old_value;
}

extern __device__ int __iAtomicExch(int *address, int val)
{
  int old_value;

  old_value = *address;
  *address = val;

  return old_value;
}

extern __device__ unsigned int
__uAtomicExch(unsigned int *address, unsigned int val)
{
  unsigned int old_value;

  old_value = *address;
  *address = val;

  return old_value;
}

extern __device__ float __fAtomicExch(float *address, float val)
{
  float old_value;

  old_value = *address;
  *address = val;

  return old_value;
}

extern __device__ int __iAtomicMin(int *address, int val)
{
  int old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  return old_value;
}

extern __device__ unsigned int
__uAtomicMin(unsigned int *address, unsigned int val)
{
  unsigned int old_value;

  old_value = *address;
  if(val < old_value)
    *address = val;

  return old_value;
}

extern __device__ int __iAtomicMax(int *address, int val)
{
  int old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  return old_value;
}

extern __device__ unsigned int
__uAtomicMax(unsigned int *address, unsigned int val)
{
  unsigned int old_value;

  old_value = *address;
  if(val > old_value)
    *address = val;

  return old_value;
}

extern __device__ unsigned int
__uAtomicInc(unsigned int *address, unsigned int val)
{
  unsigned int old_value;

  old_value = *address;

  //	((old >= val) ? 0 : (old+1))
  //	(old_value >= val) ? (*address = 0) : (*address = (old_value+1));

  if(old_value >= val)
    *address = 0;
  else
    *address = (old_value + 1);

  return old_value;
}

extern __device__ unsigned int
__uAtomicDec(unsigned int *address, unsigned int val)
{
  unsigned int old_value;

  old_value = *address;

  //	(((old == 0) | (old > val)) ? val : (old-1) )
  //	((old_value == val) | (old_value > val)) ? (*address = val) : (*address = (old_value-1));

  if((old_value == val) | (old_value > val))
    *address = val;
  else
    *address = (old_value - 1);

  return old_value;
}

extern __device__ int __iAtomicAnd(int *address, int val)
{
  int old_value;
  old_value = *address;

  *address = (old_value & val);

  return old_value;
}

extern __device__ unsigned int
__uAtomicAnd(unsigned int *address, unsigned int val)
{
  unsigned int old_value;
  old_value = *address;

  *address = (old_value & val);

  return old_value;
}

extern __device__ int __iAtomicOr(int *address, int val)
{
  int old_value;
  old_value = *address;

  *address = (old_value | val);

  return old_value;
}

extern __device__ unsigned int
__uAtomicOr(unsigned int *address, unsigned int val)
{
  unsigned int old_value;
  old_value = *address;

  *address = (old_value | val);

  return old_value;
}

extern __device__ int __iAtomicXor(int *address, int val)
{
  int old_value;
  old_value = *address;

  *address = (old_value ^ val);

  return old_value;
}
extern __device__ unsigned int
__uAtomicXor(unsigned int *address, unsigned int val)
{
  int old_value;
  old_value = *address;

  *address = (old_value ^ val);

  return old_value;
}

extern __device__ int __iAtomicCAS(int *address, int compare, int val)
{
  int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  return old_value;
}

extern __device__ unsigned int
__uAtomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  unsigned int old_value;
  old_value = *address;

  //	(old == compare ? val : old)
  //	(old_value == compare ? val : old_value)

  if(old_value == compare)
    *address = val;
  else
    *address = old_value;

  return old_value;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __device__ int atomicAdd(int *address, int val)
{
  return __iAtomicAdd(address, val);
}

static __inline__ __device__ unsigned int
atomicAdd(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, val);
}

static __inline__ __device__ int atomicSub(int *address, int val)
{
  return __iAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __device__ unsigned int
atomicSub(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd(address, (unsigned int)-(int)val);
}

static __inline__ __device__ int atomicExch(int *address, int val)
{
  return __iAtomicExch(address, val);
}

static __inline__ __device__ unsigned int
atomicExch(unsigned int *address, unsigned int val)
{
  return __uAtomicExch(address, val);
}

static __inline__ __device__ float atomicExch(float *address, float val)
{
  return __fAtomicExch(address, val);
}

static __inline__ __device__ int atomicMin(int *address, int val)
{
  return __iAtomicMin(address, val);
}

static __inline__ __device__ unsigned int
atomicMin(unsigned int *address, unsigned int val)
{
  return __uAtomicMin(address, val);
}

static __inline__ __device__ int atomicMax(int *address, int val)
{
  return __iAtomicMax(address, val);
}

static __inline__ __device__ unsigned int
atomicMax(unsigned int *address, unsigned int val)
{
  return __uAtomicMax(address, val);
}

static __inline__ __device__ unsigned int
atomicInc(unsigned int *address, unsigned int val)
{
  return __uAtomicInc(address, val);
}

static __inline__ __device__ unsigned int
atomicDec(unsigned int *address, unsigned int val)
{
  return __uAtomicDec(address, val);
}

static __inline__ __device__ int atomicAnd(int *address, int val)
{
  return __iAtomicAnd(address, val);
}

static __inline__ __device__ unsigned int
atomicAnd(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd(address, val);
}

static __inline__ __device__ int atomicOr(int *address, int val)
{
  return __iAtomicOr(address, val);
}

static __inline__ __device__ unsigned int
atomicOr(unsigned int *address, unsigned int val)
{
  return __uAtomicOr(address, val);
}

static __inline__ __device__ int atomicXor(int *address, int val)
{
  return __iAtomicXor(address, val);
}

static __inline__ __device__ unsigned int
atomicXor(unsigned int *address, unsigned int val)
{
  return __uAtomicXor(address, val);
}

static __inline__ __device__ int atomicCAS(int *address, int compare, int val)
{
  return __iAtomicCAS(address, compare, val);
}

static __inline__ __device__ unsigned int
atomicCAS(unsigned int *address, unsigned int compare, unsigned int val)
{
  return __uAtomicCAS(address, compare, val);
}

#endif /* !__SM_11_ATOMIC_FUNCTIONS_H__ */
