#ifndef _CALL_KERNEL_H
#define _CALL_KERNEL_H 1

#include <stddef.h>
#include <cstdlib>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include "vector_types.h"
#include "device_launch_parameters.h"
#include <new>

int blockGlobal;
int threadGlobal;

#define GPU_threads 2

/*ESBMC_verify_kernel()*/
typedef void *(*voidFunction_no_params)();
typedef void *(*voidFunction_one)(int *arg);
typedef void *(*voidFunction_two)(int *arg, int *arg2);
typedef void *(*voidFunction_three)(int *arg, int *arg2, int *arg3);

/*ESBMC_verify_kernel_i()*/
typedef void *(*voidFunction_two_i)(int arg, int arg2);

/*ESBMC_verify_kernel_u()*/
typedef void *(*voidFunction_one_u)(uint4 *arg);
typedef void *(*voidFunction_two_u)(unsigned int *arg, int *arg2);
typedef void *(
  *voidFunction_three_u)(unsigned arg, unsigned arg2, unsigned arg3);

/*ESBMC_verify_kernel_f()*/
typedef void *(*voidFunction_float)(float *arg1);
typedef void *(*voidFunction_ffloat)(float *arg1, float arg2);
typedef void *(*voidFunction_f5i2)(
  float *arg1,
  float *arg2,
  float *arg3,
  float *arg4,
  float *arg5,
  int arg6,
  int arg7);

/*ESBMC_verify_kernel_c()*/
typedef void *(*voidFunction_d1)(double *arg1);
typedef void *(*voidFunction_d2)(double arg1, int arg2);
typedef void *(*voidFunction_c3)(char *arg1, char *arg2, char *arg3);

/*ESBMC_verify_kernel_intt()*/
typedef void *(*voidFunction_intt)(int *arg1, int arg2);

/*ESBMC_verify_kernel_fuintt*/
typedef void *(
  *voidFunction_fuintt)(float *arg1, unsigned int arg2, unsigned int arg3);
typedef void *(*voidFunction_fuint)(float *arg1, unsigned int arg2);

/*ESBMC_verify_kernel_fuintint*/
typedef void *(
  *voidFunction_fuintint)(float *arg1, unsigned int arg2, int arg3);
typedef void *(*voidFunction_fint)(float *arg1, int arg2);

/*ESBMC_verify_kernel_three_args_iuull*/
typedef void *(*voidFunction_iuull)(
  int *arg1,
  unsigned int *arg2,
  unsigned long long int *arg3);

/*ESBMC_verify_kernel_four_args_i_ui_ull_f*/
typedef void *(*voidFunction_i_ui_ull_f)(
  int *arg1,
  unsigned int *arg2,
  unsigned long long int *arg3,
  float *arg4);

/*ESBMC_verify_kernel_ui*/
typedef void *(*voidFunction_one_ui)(
  unsigned int *arg); /* CONSERTAR ESTA DUPLICAÇÃO!! */
typedef void *(*voidFunction_ui)(unsigned int *arg);

struct arg_struct_no_params
{
  void *(*func)();
};

struct arg_struct
{
  int *a;
  int *b;
  int *c;
  void *(*func)(int *, int *, int *);
};

struct arg_struct1
{
  int *a;
  void *(*func)(int *);
};

struct arg_struct2
{
  int *a;
  int *b;
  void *(*func)(int *, int *);
};

/* ESBMC_verify_kernel() */
struct arg_struct2_i
{
  int a;
  int b;
  void *(*func)(int, int);
};

/* ESBMC_verify_kernel_u() */
struct arg_struct_u1
{
  uint4 *a;
  void *(*func)(uint4 *);
};

struct arg_struct_u2
{
  unsigned int *a;
  int *b;
  void *(*func)(unsigned int *, int *);
};

struct arg_struct_u3
{
  unsigned a;
  unsigned b;
  unsigned c;
  void *(*func)(unsigned, unsigned, unsigned);
};

/*ESBMC_verify_kernel_f()*/
struct arg_struct_float
{
  float *a;
  void *(*func)(float *);
};

struct arg_struct_ffloat
{
  float *a;
  float b;
  void *(*func)(float *, float);
};

struct arg_struct_f5i2
{
  float *a;
  float *b;
  float *c;
  float *d;
  float *e;
  int f;
  int g;
  void *(*func)(float *, float *, float *, float *, float *, int, int);
};
/*ESBMC_verify_kernel_c()*/
struct arg_struct_d1
{
  double *a;
  void *(*func)(double *);
};

struct arg_struct_d2
{
  double a;
  int b;
  void *(*func)(double, int);
};

struct arg_struct_c3
{
  char *a;
  char *b;
  char *c;
  void *(*func)(char *, char *, char *);
};

/*ESBMC_verify_kernel_intt()*/
struct arg_struct_intt
{
  int *a;
  int b;
  void *(*func)(int *, int);
};

/*ESBMC_verify_kernel_fuintt()*/

struct arg_struct_fuintt
{
  float *a;
  unsigned int b;
  unsigned int c;
  void *(*func)(float *, unsigned int, unsigned int);
};

struct arg_struct_fuint
{
  float *a;
  unsigned int b;
  void *(*func)(float *, unsigned int);
};

/*ESBMC_verify_kernel_fuintint()*/

struct arg_struct_fuintint
{
  float *a;
  unsigned int b;
  int c;
  void *(*func)(float *, unsigned int, int);
};

struct arg_struct_fint
{
  float *a;
  int b;
  void *(*func)(float *, int);
};

/*ESBMC_verify_kernel_three_args_iuull()*/
struct arg_struct_iuull
{
  int *a;
  unsigned int *b;
  unsigned long long int *c;
  void *(*func)(int *, unsigned int *, unsigned long long int *);
};

/*ESBMC_verify_kernel_four_args_i_ui_ull_f*/
struct arg_struct_i_ui_ull_f
{
  int *a;
  unsigned int *b;
  unsigned long long int *c;
  float *d;
  void *(*func)(int *, unsigned int *, unsigned long long int *, float *);
};

/*ESBMC_verify_kernel_ui()*/
struct arg_struct1_ui
{
  unsigned int *a;
  void *(*func)(unsigned int *);
};

typedef struct arg_struct_no_params Targ_no_params;
typedef struct arg_struct Targ;
typedef struct arg_struct1 Targ1;
typedef struct arg_struct2 Targ2;

typedef struct arg_struct2_i Targ2_i;

typedef struct arg_struct_u1 Targ_u1;
typedef struct arg_struct_u2 Targ_u2;
typedef struct arg_struct_u3 Targ_u3;

typedef struct arg_struct_intt Targ_intt;

typedef struct arg_struct_float Targ_float;
typedef struct arg_struct_ffloat Targ_ffloat;
typedef struct arg_struct_f5i2 Targ_f5i2;

typedef struct arg_struct_d1 Targ_d1;
typedef struct arg_struct_d2 Targ_d2;
typedef struct arg_struct_c3 Targ_c3;

typedef struct arg_struct_fuintt Targ_fuintt;
typedef struct arg_struct_fuint Targ_fuint;

typedef struct arg_struct_fuintint Targ_fuintint;
typedef struct arg_struct_fint Targ_fint;

typedef struct arg_struct_iuull Targ_iuull;
typedef struct arg_struct_i_ui_ull_f Targ_i_ui_ull_f;

typedef struct arg_struct1_ui Targ1_ui;

unsigned int n_threads;
pthread_t *threads_id;
Targ_no_params dev_no_params;
Targ dev_three;
Targ1 dev_one;
Targ2 dev_two;

Targ2_i dev_two_i;

Targ_u1 dev_one_u;
Targ_u2 dev_two_u;
Targ_u3 dev_three_u;

Targ_intt dev_intt;

Targ_float dev_float;
Targ_ffloat dev_ffloat;
Targ_f5i2 dev_f5i2;

Targ_d1 dev_d1;
Targ_d2 dev_d2;
Targ_c3 dev_c3;

Targ_fuintt dev_fuintt;
Targ_fuint dev_fuint;

Targ_fuintint dev_fuintint;
Targ_fint dev_fint;

Targ_iuull dev_three_iuull;

Targ_i_ui_ull_f dev_four_i_ui_ull_f;

Targ1_ui dev_ui;

/*ESBMC_execute_kernel()*/
void *ESBMC_execute_kernel_no_params(void *args)
{
  //ESBMC_atomic_begin();
  dev_no_params.func();
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_one(void *args)
{
  //ESBMC_atomic_begin();
  dev_one.func(dev_one.a);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_two(void *args)
{
  //ESBMC_atomic_begin();
  dev_two.func(dev_two.a, dev_two.b);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_three(void *args)
{
  //ESBMC_atomic_begin();
  dev_three.func(dev_three.a, dev_three.b, dev_three.c);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_i()*/
void *ESBMC_execute_kernel_two_i(void *args)
{
  //ESBMC_atomic_begin();
  dev_two_i.func(dev_two_i.a, dev_two_i.b);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_u()*/
void *ESBMC_execute_kernel_one_u(void *args)
{
  //ESBMC_atomic_begin();
  dev_one_u.func(dev_one_u.a);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_two_u(void *args)
{
  //ESBMC_atomic_begin();
  dev_two_u.func(dev_two_u.a, dev_two_u.b);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_three_u(void *args)
{
  //ESBMC_atomic_begin();
  dev_three_u.func(dev_three_u.a, dev_three_u.b, dev_three_u.c);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_f()*/
void *ESBMC_execute_kernel_float(void *args)
{
  //ESBMC_atomic_begin();
  dev_float.func(dev_float.a);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_ffloat(void *args)
{
  //ESBMC_atomic_begin();
  dev_ffloat.func(dev_ffloat.a, dev_ffloat.b);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_f5i2(void *args)
{
  //ESBMC_atomic_begin();
  dev_f5i2.func(
    dev_f5i2.a,
    dev_f5i2.b,
    dev_f5i2.c,
    dev_f5i2.d,
    dev_f5i2.e,
    dev_f5i2.f,
    dev_f5i2.g);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_c()*/
void *ESBMC_execute_kernel_d1(void *args)
{
  //ESBMC_atomic_begin();
  dev_d1.func(dev_d1.a);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_d2(void *args)
{
  //ESBMC_atomic_begin();
  dev_d2.func(dev_d2.a, dev_d2.b);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_c3(void *args)
{
  //ESBMC_atomic_begin();
  dev_c3.func(dev_c3.a, dev_c3.b, dev_c3.c);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_intt()*/
void *ESBMC_execute_kernel_intt(void *args)
{
  //ESBMC_atomic_begin();
  dev_intt.func(dev_intt.a, dev_intt.b);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_fuintt()*/
void *ESBMC_execute_kernel_fuintt(void *args)
{
  //ESBMC_atomic_begin();
  dev_fuintt.func(dev_fuintt.a, dev_fuintt.b, dev_fuintt.c);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_fuint(void *args)
{
  //ESBMC_atomic_begin();
  dev_fuint.func(dev_fuint.a, dev_fuint.b);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_fuintint()*/
void *ESBMC_execute_kernel_fuintint(void *args)
{
  //ESBMC_atomic_begin();
  dev_fuintint.func(dev_fuintint.a, dev_fuintint.b, dev_fuintint.c);
  //ESBMC_atomic_end();
  return NULL;
}

void *ESBMC_execute_kernel_fint(void *args)
{
  //ESBMC_atomic_begin();
  dev_fint.func(dev_fint.a, dev_fint.b);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_three_args_iuull()*/
void *ESBMC_execute_kernel_three_iuull(void *args)
{
  //ESBMC_atomic_begin();
  dev_three_iuull.func(dev_three_iuull.a, dev_three_iuull.b, dev_three_iuull.c);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_four_args_iuull()*/
void *ESBMC_execute_kernel_four_i_ui_ull_f(void *args)
{
  //ESBMC_atomic_begin();
  dev_four_i_ui_ull_f.func(
    dev_four_i_ui_ull_f.a,
    dev_four_i_ui_ull_f.b,
    dev_four_i_ui_ull_f.c,
    dev_four_i_ui_ull_f.d);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_execute_kernel_three_args_ui()*/
void *ESBMC_execute_kernel_ui(void *args)
{
  //ESBMC_atomic_begin();
  dev_ui.func(dev_ui.a);
  //ESBMC_atomic_end();
  return NULL;
}

/*ESBMC_verify_kernel()*/
void ESBMC_verify_kernel_no_params(void *(*kernel)(), int blocks, int threads)
{
  __ESBMC_atomic_begin();
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  dev_no_params.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_no_params, NULL);
    i++;
  }
  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_with_one_arg(
  void *(*kernel)(int *),
  int blocks,
  int threads,
  void *arg1)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_one.a = (int*) malloc(n_threads * sizeof(int));
  dev_one.a = (int *)malloc(GPU_threads * sizeof(int));
  dev_one.a = (int *)arg1;
  dev_one.func = kernel;

  blockGlobal = blocks;
  threadGlobal = threads;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_one, NULL);
    //		assert(0);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_with_two_args(
  void *(*kernel)(int *, int *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_two.a = (int*) malloc(n_threads * sizeof(int));
  dev_two.a = (int *)malloc(GPU_threads * sizeof(int));
  //dev_two.b = (int*) malloc(n_threads * sizeof(int));
  dev_two.b = (int *)malloc(GPU_threads * sizeof(int));
  dev_two.a = (int *)arg1;
  dev_two.b = (int *)arg2;
  dev_two.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_two, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_with_three_args(
  void *(*kernel)(int *, int *, int *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2,
  void *arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_three.a = (int*) malloc(n_threads * sizeof(int));
  dev_three.a = (int *)malloc(GPU_threads * sizeof(int));
  //dev_three.b = (int*) malloc(n_threads * sizeof(int));
  dev_three.b = (int *)malloc(GPU_threads * sizeof(int));
  //dev_three.c = (int*) malloc(n_threads * sizeof(int));
  dev_three.c = (int *)malloc(GPU_threads * sizeof(int));

  dev_three.a = (int *)arg1;
  dev_three.b = (int *)arg2;
  dev_three.c = (int *)arg3;
  dev_three.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_three, NULL);
    //assert(0);
    i++;
  }
  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_i()*/
void ESBMC_verify_kernel_with_two_args_i(
  void *(*kernel)(int, int),
  int blocks,
  int threads,
  int arg1,
  int arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  dev_two_i.a = arg1;
  dev_two_i.b = arg2;
  dev_two_i.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_two_i, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_u()*/
void ESBMC_verify_kernel_with_one_args_u(
  void *(*kernel)(uint4 *),
  int blocks,
  int threads,
  void *arg1)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_one_u.a = (uint4*) malloc(n_threads * sizeof(uint4));
  dev_one_u.a = (uint4 *)malloc(GPU_threads * sizeof(uint4));

  dev_one_u.a = (uint4 *)arg1;
  dev_one_u.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_one_u, NULL);
    //assert(0);
    i++;
  }
  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_with_two_args_u(
  void *(*kernel)(unsigned int *, int *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_two_u.a = (unsigned int*) malloc(n_threads * sizeof(unsigned int));
  dev_two_u.a = (unsigned int *)malloc(GPU_threads * sizeof(unsigned int));
  //dev_two_u.b = (int*) malloc(n_threads * sizeof(int));
  dev_two_u.b = (int *)malloc(GPU_threads * sizeof(int));
  dev_two_u.a = (unsigned int *)arg1;
  dev_two_u.b = (int *)arg2;
  dev_two_u.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_two_u, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_with_three_args_u(
  void *(*kernel)(unsigned, unsigned, unsigned),
  int blocks,
  int threads,
  unsigned arg1,
  unsigned arg2,
  unsigned arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  dev_three_u.a = arg1;
  dev_three_u.b = arg2;
  dev_three_u.c = arg3;
  dev_three_u.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_three_u, NULL);
    //assert(0);
    i++;
  }
  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_f()*/
void ESBMC_verify_kernel_float(
  void *(*kernel)(float *),
  int blocks,
  int threads,
  void *arg1)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_float.a = (float*) malloc(n_threads * sizeof(float));
  dev_float.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_float.a = (float *)arg1;
  dev_float.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_float, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_ffloat(
  void *(*kernel)(float *, float),
  int blocks,
  int threads,
  void *arg1,
  float arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_ffloat.a = (float*) malloc(n_threads * sizeof(float));
  dev_ffloat.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_ffloat.a = (float *)arg1;
  dev_ffloat.b = arg2;
  dev_ffloat.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_ffloat, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_f5i2(
  void *(*kernel)(float *, float *, float *, float *, float *, int, int),
  int blocks,
  int threads,
  void *arg1,
  void *arg2,
  void *arg3,
  void *arg4,
  void *arg5,
  int arg6,
  int arg7)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_f5i2.a = (float*) malloc(GPU_threads * sizeof(float));
  dev_f5i2.a = (float *)malloc(GPU_threads * sizeof(float));
  //dev_f5i2.b = (float*) malloc(GPU_threads * sizeof(float));
  dev_f5i2.b = (float *)malloc(GPU_threads * sizeof(float));
  //dev_f5i2.c = (float*) malloc(GPU_threads * sizeof(float));
  dev_f5i2.c = (float *)malloc(GPU_threads * sizeof(float));
  //dev_f5i2.d = (float*) malloc(GPU_threads * sizeof(float));
  dev_f5i2.d = (float *)malloc(GPU_threads * sizeof(float));
  //dev_f5i2.e = (float*) malloc(GPU_threads * sizeof(float));
  dev_f5i2.e = (float *)malloc(GPU_threads * sizeof(float));

  dev_f5i2.a = (float *)arg1;
  dev_f5i2.b = (float *)arg2;
  dev_f5i2.c = (float *)arg3;
  dev_f5i2.d = (float *)arg4;
  dev_f5i2.e = (float *)arg5;
  dev_f5i2.f = arg6;
  dev_f5i2.g = arg7;
  dev_f5i2.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_f5i2, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_c()*/
void ESBMC_verify_kernel_d1(
  void *(*kernel)(double *),
  int blocks,
  int threads,
  void *arg1)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  dev_d1.a = (double *)malloc(GPU_threads * sizeof(double));

  dev_d1.a = (double *)arg1;

  dev_d1.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_d1, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_d2(
  void *(*kernel)(double, int),
  int blocks,
  int threads,
  double arg1,
  int arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_intt.a = (int*) malloc(n_threads * sizeof(int));
  dev_d2.a = arg1;
  dev_d2.b = arg2;
  dev_d2.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_d2, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel_c3(
  void *(*kernel)(char *, char *, char *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2,
  void *arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  dev_c3.a = (char *)malloc(GPU_threads * sizeof(float));
  dev_c3.b = (char *)malloc(GPU_threads * sizeof(float));
  dev_c3.c = (char *)malloc(GPU_threads * sizeof(float));

  dev_c3.a = (char *)arg1;
  dev_c3.b = (char *)arg2;
  dev_c3.c = (char *)arg3;
  dev_c3.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_c3, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_intt()*/
void ESBMC_verify_kernel__intt(
  void *(*kernel)(int *, int),
  int blocks,
  int threads,
  void *arg1,
  int arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_intt.a = (int*) malloc(n_threads * sizeof(int));
  dev_intt.a = (int *)malloc(GPU_threads * sizeof(int));

  dev_intt.a = (int *)arg1;
  dev_intt.b = arg2;
  dev_intt.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_intt, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_fuintt()*/
void ESBMC_verify_kernel__fuintt(
  void *(*kernel)(float *, unsigned int, unsigned int),
  int blocks,
  int threads,
  void *arg1,
  unsigned int arg2,
  unsigned int arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_fuintt.a = (float*) malloc(n_threads * sizeof(float));
  dev_fuintt.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_fuintt.a = (float *)arg1;
  dev_fuintt.b = arg2;
  dev_fuintt.c = arg3;
  dev_fuintt.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_fuintt, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel__fuint(
  void *(*kernel)(float *, unsigned int),
  int blocks,
  int threads,
  void *arg1,
  unsigned int arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_fuint.a = (float*) malloc(n_threads * sizeof(float));
  dev_fuint.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_fuint.a = (float *)arg1;
  dev_fuint.b = arg2;
  dev_fuint.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_fuint, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_fuintint()*/
void ESBMC_verify_kernel__fuintint(
  void *(*kernel)(float *, unsigned int, int),
  int blocks,
  int threads,
  void *arg1,
  unsigned int arg2,
  int arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_fuintint.a = (float*) malloc(n_threads * sizeof(float));
  dev_fuintint.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_fuintint.a = (float *)arg1;
  dev_fuintint.b = arg2;
  dev_fuintint.c = arg3;
  dev_fuintint.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_fuintint, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

void ESBMC_verify_kernel__fint(
  void *(*kernel)(float *, int),
  int blocks,
  int threads,
  void *arg1,
  int arg2)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_fint.a = (float*) malloc(n_threads * sizeof(float));
  dev_fint.a = (float *)malloc(GPU_threads * sizeof(float));

  dev_fint.a = (float *)arg1;
  dev_fint.b = arg2;
  dev_fint.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_fint, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_three_args_iuull()*/
void ESBMC_verify_kernel_with_three_args_iuull(
  void *(*kernel)(int *, unsigned int *, unsigned long long int *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2,
  void *arg3)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_three_iuull.a = (int*) malloc(n_threads * sizeof(int));
  dev_three_iuull.a = (int *)malloc(GPU_threads * sizeof(int));
  //dev_three_iuull.b = (unsigned int*) malloc(n_threads * sizeof(unsigned int));
  dev_three_iuull.b =
    (unsigned int *)malloc(GPU_threads * sizeof(unsigned int));
  //dev_three_iuull.c = (unsigned long long int*) malloc(n_threads * sizeof(unsigned long long int));
  dev_three_iuull.c = (unsigned long long int *)malloc(
    GPU_threads * sizeof(unsigned long long int));

  dev_three_iuull.a = (int *)arg1;
  dev_three_iuull.b = (unsigned int *)arg2;
  dev_three_iuull.c = (unsigned long long int *)arg3;
  dev_three_iuull.func = kernel;

  unsigned int n_threads = blocks * threads;
  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(
      &threads_id[i], NULL, ESBMC_execute_kernel_three_iuull, NULL);
    i++;
  }
  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_four_args_i_ui_ull_f()*/
void ESBMC_verify_kernel_with_four__args_i_ui_ull_f(
  void *(*kernel)(int *, unsigned int *, unsigned long long int *, float *),
  int blocks,
  int threads,
  void *arg1,
  void *arg2,
  void *arg3,
  void *arg4)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_four_i_ui_ull_f.a = (int*) malloc(n_threads * sizeof(int));
  dev_four_i_ui_ull_f.a = (int *)malloc(GPU_threads * sizeof(int));
  //dev_four_i_ui_ull_f.b = (unsigned int*) malloc(n_threads * sizeof(unsigned int));
  dev_four_i_ui_ull_f.b =
    (unsigned int *)malloc(GPU_threads * sizeof(unsigned int));
  //dev_four_i_ui_ull_f.c = (unsigned long long int*) malloc(n_threads * sizeof(unsigned long long int));
  dev_four_i_ui_ull_f.c = (unsigned long long int *)malloc(
    GPU_threads * sizeof(unsigned long long int));
  //dev_four_i_ui_ull_f.d = (float*) malloc(n_threads * sizeof(float));
  dev_four_i_ui_ull_f.d = (float *)malloc(GPU_threads * sizeof(float));

  dev_four_i_ui_ull_f.a = (int *)arg1;
  dev_four_i_ui_ull_f.b = (unsigned int *)arg2;
  dev_four_i_ui_ull_f.c = (unsigned long long int *)arg3;
  dev_four_i_ui_ull_f.d = (float *)arg4;
  dev_four_i_ui_ull_f.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(
      &threads_id[i], NULL, ESBMC_execute_kernel_four_i_ui_ull_f, NULL);
    i++;
  }
  __ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_three_args_ui()*/
void ESBMC_verify_kernel_one_ui(
  void *(*kernel)(unsigned int *),
  int blocks,
  int threads,
  void *arg1)
{
  __ESBMC_atomic_begin();
  //n_threads = blocks * threads;
  threads_id = (pthread_t *)malloc(GPU_threads * sizeof(pthread_t));
  //threads_id = (pthread_t *) malloc(2 * sizeof(pthread_t));

  //dev_ui.a = (unsigned int*) malloc(n_threads * sizeof(unsigned int));
  dev_ui.a = (unsigned int *)malloc(GPU_threads * sizeof(unsigned int));
  dev_ui.a = (unsigned int *)arg1;
  dev_ui.func = kernel;

  int i = 0, tmp;
  assignIndexes();
  while(i < GPU_threads)
  {
    //while (i < 2) {
    pthread_create(&threads_id[i], NULL, ESBMC_execute_kernel_ui, NULL);
    i++;
  }

  __ESBMC_atomic_end();
}

/************ TEMPLATES **********/
/*ESBMC_verify_kernel()*/
template <class RET, class BLOCK, class THREAD>
void ESBMC_verify_kernel(RET *kernel, BLOCK blocks, THREAD threads)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_no_params(
    (voidFunction_no_params)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1>
void ESBMC_verify_kernel(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_one_arg(
    (voidFunction_one)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_two_args(
    (voidFunction_two)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_three_args(
    (voidFunction_three)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2,
    (void *)arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_i*/
template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_i(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_two_args_i(
    (voidFunction_two_i)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_u()*/
template <class RET, class BLOCK, class THREAD, class T1>
void ESBMC_verify_kernel_u(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_one_args_u(
    (voidFunction_one_u)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    arg);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_u(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_two_args_u(
    (voidFunction_two_u)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel_u(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_three_args_u(
    (voidFunction_three_u)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    arg,
    arg2,
    arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_f()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_f(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_ffloat(
    (voidFunction_ffloat)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1>
void ESBMC_verify_kernel_f(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_float(
    (voidFunction_float)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}
template <
  class RET,
  class BLOCK,
  class THREAD,
  class T1,
  class T2,
  class T3,
  class T4,
  class T5,
  class T6,
  class T7>
void ESBMC_verify_kernel_f(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3,
  T4 arg4,
  T5 arg5,
  T6 arg6,
  T7 arg7)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_f5i2(
    (voidFunction_f5i2)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2,
    (void *)arg3,
    (void *)arg4,
    (void *)arg5,
    arg6,
    arg7);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_c()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel_c(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_c3(
    (voidFunction_c3)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2,
    (void *)arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1>
void ESBMC_verify_kernel_c(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_d1(
    (voidFunction_d1)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_c(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_d2(
    (voidFunction_d2)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_intt()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_intt(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel__intt(
    (voidFunction_intt)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_fuintt()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel_fuintt(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel__fuintt(
    (voidFunction_fuintt)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2,
    arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_fuintt(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel__fuint(
    (voidFunction_fuint)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_fuintint()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel_fuintint(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel__fuintint(
    (voidFunction_fuintint)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2,
    arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

template <class RET, class BLOCK, class THREAD, class T1, class T2>
void ESBMC_verify_kernel_fuintint(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel__fint(
    (voidFunction_fint)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    arg2);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel__three_args_iuull()*/
template <class RET, class BLOCK, class THREAD, class T1, class T2, class T3>
void ESBMC_verify_kernel_three_args_iuull(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_three_args_iuull(
    (voidFunction_iuull)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2,
    (void *)arg3);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
/*ESBMC_verify_kernel__four_args_iuull()*/
template <
  class RET,
  class BLOCK,
  class THREAD,
  class T1,
  class T2,
  class T3,
  class T4>
void ESBMC_verify_kernel_four(
  RET *kernel,
  BLOCK blocks,
  THREAD threads,
  T1 arg,
  T2 arg2,
  T3 arg3,
  T4 arg4)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_with_four__args_i_ui_ull_f(
    (voidFunction_i_ui_ull_f)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg,
    (void *)arg2,
    (void *)arg3,
    (void *)arg4);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

/*ESBMC_verify_kernel_ui()*/
template <class RET, class BLOCK, class THREAD, class T1>
void ESBMC_verify_kernel_ui(RET *kernel, BLOCK blocks, THREAD threads, T1 arg)
{
  //ESBMC_atomic_begin();
  gridDim = dim3(blocks);
  blockDim = dim3(threads);

  ESBMC_verify_kernel_one_ui(
    (voidFunction_one_ui)kernel,
    gridDim.x * gridDim.y * gridDim.z,
    blockDim.x * blockDim.y * blockDim.z,
    (void *)arg);

  int i = 0;
  for(i = 0; i < GPU_threads; i++)
    //for (i = 0; i < 2; i++)
    pthread_join(threads_id[i], NULL);

  //ESBMC_atomic_end();
}

#endif /*call_kernel*/
