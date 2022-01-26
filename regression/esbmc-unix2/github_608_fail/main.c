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

extern void __assert_fail(
  const char *__assertion,
  const char *__file,
  unsigned int __line,
  const char *__function) __attribute__((__nothrow__, __leaf__))
__attribute__((__noreturn__));
void reach_error()
{
  __assert_fail(
    "0", "chl-simpl-str-symm.wvr.c", 21, __extension__ __PRETTY_FUNCTION__);
}
extern int pthread_create(
  pthread_t *__restrict __newthread,
  const pthread_attr_t *__restrict __attr,
  void *(*__start_routine)(void *),
  void *__restrict __arg) __attribute__((__nothrow__))
__attribute__((__nonnull__(1, 3)));
extern int pthread_join(pthread_t __th, void **__thread_return);

extern int __VERIFIER_nondet_int(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern void __VERIFIER_atomic_begin();
extern void __VERIFIER_atomic_end();

extern void abort(void);
void assume_abort_if_not(int cond)
{
  if(!cond)
  {
    abort();
  }
}

int length_0, to_int_1, length_2, to_int_3, result_4, result_5; // 0

int minus(int a, int b);

void *thread1()
{
  result_4 = length_0 == 0 ? (length_2 == 0 ? 0 : 1)
                           : (length_2 == 0 ? -1 : minus(to_int_1, to_int_3));
  return 0;
}

void *thread2()
{
  result_5 = length_2 == 0
               ? (length_0 == 0 ? 0 : 1)
               : (length_0 == 0 ? -1 : minus(to_int_3, to_int_1)); //7: 1
  return 0;
}

int main()
{
  pthread_t t1, t2;

  // initialize global variables
  length_0 = __VERIFIER_nondet_int(); // 1: -2147483648
  to_int_1 = __VERIFIER_nondet_int(); // 2: 2147483647
  length_2 = __VERIFIER_nondet_int(); // 3: 0
  to_int_3 = __VERIFIER_nondet_int(); // 4: 2147483647

  // main method
  pthread_create(&t1, 0, thread1, 0); // 5:
  pthread_create(&t2, 0, thread2, 0); // 6:
  // Removing the following line should make the program to fail
  //pthread_join(t1, 0);
  pthread_join(t2, 0);

  assume_abort_if_not(!(result_4 < 0 && result_5 > 0));
  assume_abort_if_not(!(result_4 > 0 && result_5 < 0));
  assume_abort_if_not(!(result_4 == 0 && result_5 == 0));
  reach_error();

  return 0;
}

int minus(int a, int b)
{
  assume_abort_if_not(b <= 0 || a >= b - 2147483648);
  assume_abort_if_not(b >= 0 || a <= b + 2147483647);
  return a - b;
}
