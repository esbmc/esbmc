extern void abort(void);
#include <assert.h>
void reach_error() { assert(0); }
extern void abort(void);
void assume_abort_if_not(int cond) {
  if(!cond) {abort();}
}


extern  int __VERIFIER_nondet_int();

void __VERIFIER_assert(int cond) {
  if (!(cond)) {
    ERROR: {reach_error();abort();}
  }
  return;
}

#include <pthread.h>

int value;

void __VERIFIER_atomic_check_a()
{
  //__VERIFIER_atomic_begin();
  value = 42;
  if(value != 42) reach_error();
  //__VERIFIER_atomic_end();
}

void __VERIFIER_atomic_check_b()
{
  //__VERIFIER_atomic_begin();
  value = 100;
  if(value != 100) reach_error();
  // __VERIFIER_atomic_end();
}

void* dec_a(void* arg)
{
  __VERIFIER_atomic_check_a();

  return 0;
}

void* dec_b(void* arg)
{
  __VERIFIER_atomic_check_b();
  return 0;
}

unsigned start()
{
  pthread_t t1, t2;

  pthread_create(&t1, 0, dec_a, 0);
  pthread_create(&t2, 0, dec_b, 0);

  pthread_join(t1, 0);
  pthread_join(t2, 0);

  return 0;
}


int main()
{
  start();
  return 0;
}

