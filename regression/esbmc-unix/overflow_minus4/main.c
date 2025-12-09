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
void reach_error() { __assert_fail("0", "chl-collitem-subst.wvr.c", 21, __extension__ __PRETTY_FUNCTION__); }
extern int pthread_create (pthread_t *__restrict __newthread,
      const pthread_attr_t *__restrict __attr,
      void *(*__start_routine) (void *),
      void *__restrict __arg) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 3)));
extern int pthread_join (pthread_t __th, void **__thread_return);

extern int  __VERIFIER_nondet_int(void);
extern _Bool __VERIFIER_nondet_bool(void);
extern void __VERIFIER_atomic_begin(void);
extern void __VERIFIER_atomic_end(void);

extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}

int cardSet_0, cardRarity_1, cardId_2, cardType_3, cardSet_4, cardRarity_5, cardId_6, cardType_7, cardSet_8, cardRarity_9, cardId_10, cardType_11, result_12, result_13, result_14;

int minus(int a, int b);

void* thread1(void* _argptr) {
  __VERIFIER_atomic_begin();
  result_12 = minus(cardSet_0, cardSet_4);
  result_12 = result_12 == 0 ? minus(cardRarity_1, cardRarity_5) : result_12;
  result_12 = result_12 == 0 ? minus(cardId_2, cardId_6) : result_12;
  result_12 = result_12 == 0 ? minus(cardType_3, cardType_7) : result_12;
  __VERIFIER_atomic_end();

  return 0;
}

void* thread2(void* _argptr) {
  __VERIFIER_atomic_begin();
  result_13 = minus(cardSet_0, cardSet_8);
  result_13 = result_13 == 0 ? minus(cardRarity_1, cardRarity_9) : result_13;
  result_13 = result_13 == 0 ? minus(cardId_2, cardId_10) : result_13;
  result_13 = result_13 == 0 ? minus(cardType_3, cardType_11) : result_13;
  __VERIFIER_atomic_end();

  return 0;
}

void* thread3(void* _argptr) {
  __VERIFIER_atomic_begin();
  result_14 = minus(cardSet_4, cardSet_8);
  result_14 = result_14 == 0 ? minus(cardRarity_5, cardRarity_9) : result_14;
  result_14 = result_14 == 0 ? minus(cardId_6, cardId_10) : result_14;
  result_14 = result_14 == 0 ? minus(cardType_7, cardType_11) : result_14;
  __VERIFIER_atomic_end();

  return 0;
}

int main() {
  pthread_t t1, t2, t3;
  
  cardSet_0 = __VERIFIER_nondet_int();
  cardRarity_1 = __VERIFIER_nondet_int();
  cardId_2 = __VERIFIER_nondet_int();
  cardType_3 = __VERIFIER_nondet_int();
  cardSet_4 = __VERIFIER_nondet_int();
  cardRarity_5 = __VERIFIER_nondet_int();
  cardId_6 = __VERIFIER_nondet_int();
  cardType_7 = __VERIFIER_nondet_int();
  cardSet_8 = __VERIFIER_nondet_int();
  cardRarity_9 = __VERIFIER_nondet_int();
  cardId_10 = __VERIFIER_nondet_int();
  cardType_11 = __VERIFIER_nondet_int();
  
  // main method
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  pthread_create(&t3, 0, thread3, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  pthread_join(t3, 0);

  assume_abort_if_not(result_12 == 0);
  assume_abort_if_not((result_13 > 0) != (result_14 > 0) || (result_13 < 0) != (result_14 < 0));
  reach_error();

  return 0;
}

int minus(int a, int b) {
  assume_abort_if_not(b <= 0 || a >= b - 2147483648);
  assume_abort_if_not(b >= 0 || a <= b + 2147483647);
  return a - b;
}