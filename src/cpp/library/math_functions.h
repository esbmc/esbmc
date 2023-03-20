
#include <stddef.h>
#include <stdio.h>
#include <driver_types.h>
#include "definitions.h"
#include <device_functions_decls.h>
//#include "../../ansi-c/library/intrinsics.h"

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923132169164 // Pi/2
#define NEPERIANO 2.718281828
#define PREC 1e-16

//#define __device__ void
//#define __host__ void
//#define __RETURN_TYPE void
//#define __DEVICE_FUNCTIONS_DECL__ void

//###############################################################
// Signature

//Module Single Precision Mathematical Functions

__device__ float exp10f(float x);
/*__device__ float 	exp2f (float x);
__device__ float 	expf (float x);
*/
__device__ float powf_gpu(
  float x,
  float y); //powf_gpu para não confundir com pow definida por c++ em math.h

//Module Double Precision Mathematical Functions
/*__host__ __device__  double 	exp ( double  x );
__host__ __device__  double 	exp10 ( double  x );
__host__ __device__  double 	exp2 ( double  x );
__host__ __device__  double 	expm1 ( double  x );
__host__ __device__  double 	frexp ( double  x, int* nptr );
*/

//Module Single Precision Intrinsics
//__DEVICE_FUNCTIONS_DECL__
float __exp10f(float x);
//__DEVICE_FUNCTIONS_DECL__ float 	__expf ( float  x );
//__DEVICE_FUNCTIONS_DECL__
float __powf_gpu(float x, float y);

//Module Double Precision Intrinsics
/*__device__  double __ddiv_rd ( double  x, double  y );
__device__  double __ddiv_rn ( double  x, double  y );
__device__  double __ddiv_ru ( double  x, double  y );
__device__  double __ddiv_rz ( double  x, double  y );
__device__  double __drcp_rd ( double  x );
__device__  double __drcp_rn ( double  x );
__device__  double __drcp_ru ( double  x );
__device__  double __drcp_rz ( double  x );
__device__  double __dsqrt_rd ( double  x );
__device__  double __dsqrt_rn ( double  x );
__device__  double __dsqrt_ru ( double  x );
__device__  double __dsqrt_rz ( double  x );
*/

//###############################################################
//Module Single Precision Mathematical Functions

__device__ float exp10f(float x)
{
  return __exp10f(x);
}
/*
__device__ float 	exp2f (float x){

	return __powf(2,x);

}

__device__ float 	expf (float x){

	return __expf(x);

}

__device__ float 	expm1f (float x){

	return (__expf(x) - 1);

}


__device__ float 	frexpf (float x, int *nptr){

	float result;

	return result;
}


__device__ float 	ldexpf (float x, int exp){

	return x*__pow(2,exp);

}
*/

__device__ float powf_gpu(float a, float b)
{
  return __powf_gpu(a, b);
}

//###############################################################
//Module Double Precision Mathematical Functions
/*
__host__ ​ __device__ ​ double 	exp ( double  x )
__host__ ​ __device__ ​ double 	exp10 ( double  x )
__host__ ​ __device__ ​ double 	exp2 ( double  x )
__host__ ​ __device__ ​ double 	expm1 ( double  x )

__host__ ​ __device__ ​ double 	frexp ( double  x, int* nptr )

__host__ ​ __device__ ​ double 	ldexp ( double  x, int  exp )

*/

//###############################################################
//Module Single Precision Intrinsics

//__DEVICE_FUNCTIONS_DECL__
float __exp10f(float x)
{
  return __powf_gpu(10, x);
}
/*
__DEVICE_FUNCTIONS_DECL__ float 	__expf ( float  x ){

	return __powf(NEPERIANO,x);

}
*/

float __powf_gpu(float x, float y)
{
  int result = 1;

  if(y == 0)
    return result;

  if(y < 0)
    return 1 /
           __powf_gpu(x, -y); /// verificar se funciona, por causa do pow de C++
  else
  {
    for(int i = 0; i < y; ++i)
      result *= x;

    return result;
  }
}

//###############################################################
//Module Double Precision Intrinsics
/*
__device__  double __ddiv_rd ( double  x, double  y ){

	return  (int)(x/y);

}
*/

//###############################################################
//Module Integer Intrinsics

//__DEVICE_FUNCTIONS_DECL__ unsigned int 	__brev ( unsigned int  x );
//__DEVICE_FUNCTIONS_DECL__
int __mul24(int x, int y);
//__DEVICE_FUNCTIONS_DECL__
unsigned int __umul24(unsigned int x, unsigned int y);

//###############################################################
//Module Type Casting Intrinsics

//__DEVICE_FUNCTIONS_DECL__ int __double2int_rz ( double );

//###############################################################
//Module Integer Intrinsics

int __mul24(int x, int y)
{
  int cx;
  int cy;
  cx = x & 0x0000000000111111;
  cy = y & 0x0000000000111111;
  return cx * cy;
}

unsigned int __umul24(unsigned int x, unsigned int y)
{
  unsigned cx;
  unsigned cy;
  cx = x & 0x0000000000111111;
  cy = y & 0x0000000000111111;
  return cx * cy;
}
