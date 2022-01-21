// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2021 F. Schuessele <schuessf@informatik.uni-freiburg.de>
// SPDX-FileCopyrightText: 2021 D. Klumpp <klumpp@informatik.uni-freiburg.de>
//
// SPDX-License-Identifier: LicenseRef-BSD-3-Clause-Attribution-Vandikas

typedef unsigned long int pthread_t;

union pthread_attr_t
{
  char __size[36];
  long int __align;
};
typedef union pthread_attr_t pthread_attr_t;

extern void __assert_fail(const char *__assertion, const char *__file,
      unsigned int __line, const char *__function)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));
void reach_error() { __assert_fail("0", "parallel-misc-3.wvr.c", 21, __extension__ __PRETTY_FUNCTION__); }
extern int pthread_create (pthread_t *__restrict __newthread,
      const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));
extern int pthread_join (pthread_t __th, void **__thread_return);

extern int   __VERIFIER_nondet_int(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern void  __VERIFIER_atomic_begin();
extern void  __VERIFIER_atomic_end();

extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

int pos;
_Bool d1, d2, g1, g2;

void* thread1() {
  while (g1) {
    if (d1) {
      __VERIFIER_atomic_begin();
      pos++;
      __VERIFIER_atomic_end();
    } else {
      __VERIFIER_atomic_begin();
      pos--;
      __VERIFIER_atomic_end();
    }
    d1 = !d1;
    if (d1) {
      if (__VERIFIER_nondet_bool()) {
        g1 = 0;
      }
    }
  }

  return 0;
}

void* thread2() {
  while (g2) {
    if (d2) {
      __VERIFIER_atomic_begin();
      pos = ( pos + 2 );
      __VERIFIER_atomic_end();
    } else {
      __VERIFIER_atomic_begin();
      pos = ( pos - 2 );
      __VERIFIER_atomic_end();
    }
    d2 = !d2;
    if (d2) {
      if (__VERIFIER_nondet_bool()) {
        g2 = 0;
      }
    }
  }

  return 0;
}

int main() {
  pthread_t t1, t2;

  // initialize global variables
  d1 = 1;
  d2 = 1;
  g1 = 1;
  g2 = 1;

  // main method
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);

  assume_abort_if_not(pos != 0);
  reach_error();

  return 0;
}
