#pragma once

#include <driver_types.h>

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923132169164 // Pi/2
#define NEPERIANO 2.718281828
#define PREC 1e-16

//Module Single Precision Mathematical Functions

__device__ float exp10f(float x);

__device__ float powf_gpu(
  float x,
  float y); //powf_gpu para não confundir com pow definida por c++ em math.h

float __exp10f(float x);

float __powf_gpu(float x, float y);

__device__ float exp10f(float x)
{
  return __exp10f(x);
}

__device__ float powf_gpu(float a, float b)
{
  return __powf_gpu(a, b);
}

float __exp10f(float x)
{
  return __powf_gpu(10, x);
}

float __powf_gpu(float x, float y)
{
  int result = 1;

  if (y == 0)
    return result;

  if (y < 0)
    return 1 / __powf_gpu(x, -y);
  else
  {
    for (int i = 0; i < y; ++i)
      result *= x;

    return result;
  }
}

int __mul24(int x, int y);

unsigned int __umul24(unsigned int x, unsigned int y);

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
