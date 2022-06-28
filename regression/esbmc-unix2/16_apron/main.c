// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2005-2021 University of Tartu & Technische Universität München
//
// SPDX-License-Identifier: MIT

#include <assert.h>
extern void abort(void);
void reach_error()
{
  assert(0);
} //S6
void __VERIFIER_assert(int cond)
{
  if(!(cond))
  {
  ERROR:
  {
    reach_error();
    abort();
  }
  }
}
void assume_abort_if_not(int cond)
{
  if(!cond)
  {
    abort();
  }
}

extern int __VERIFIER_nondet_int();

#include <pthread.h>

int plus(int a, int b);

int g = 0;
int h = 0;
int i = 0;
pthread_mutex_t A = PTHREAD_MUTEX_INITIALIZER; // h <= g
pthread_mutex_t B = PTHREAD_MUTEX_INITIALIZER; // h == g
pthread_mutex_t C = PTHREAD_MUTEX_INITIALIZER;

void *t_fun(void *arg)
{
  int a = __VERIFIER_nondet_int(); //rand  //S2  -130129
  int y = __VERIFIER_nondet_int(); //rand
  int z = __VERIFIER_nondet_int(); //rand
  if(a < 1000)
  { // avoid overflow
    pthread_mutex_lock(&C);
    pthread_mutex_lock(&A);
    a = g; //S3 2147483633
    y = h; //S4  2147483646
    __VERIFIER_assert(y <= a);
    pthread_mutex_lock(&B);
    __VERIFIER_assert(a == y);
    pthread_mutex_unlock(&B);
    i = plus(a, 31);
    z = i;
    __VERIFIER_assert(z >= a);
    pthread_mutex_unlock(&A);
    pthread_mutex_unlock(&C);
  }
  return NULL;
}

int main(void)
{
  int x = __VERIFIER_nondet_int(); //rand  //S1 2147483633
  if(x > -1000)
  { // avoid underflow
    pthread_t id;
    pthread_create(&id, NULL, t_fun, NULL);

    pthread_mutex_lock(&B);
    pthread_mutex_lock(&A);
    i = 11;
    g = x; //S2  g = 2147483631
    h = plus(x, -17);
    pthread_mutex_unlock(&A);
    pthread_mutex_lock(&A);
    h = x; //S3 h = 2147483631
    pthread_mutex_unlock(&A);
    pthread_mutex_unlock(&B);
#if 1
    pthread_mutex_lock(&C);
    i = 3;
    pthread_mutex_unlock(&C);
#endif
  }
  return 0;
}

int plus(int a, int b)
{
  assume_abort_if_not(b >= 0 || a >= -2147483648 - b);
  assume_abort_if_not(b <= 0 || a <= 2147483647 - b);
  return a + b;
}
