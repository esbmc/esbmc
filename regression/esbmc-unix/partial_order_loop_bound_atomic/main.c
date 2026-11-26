/* #10990: a thread halted by an unwinding bound inside an atomic section
 * never leaves it, so main never sees x == 1. */
#include <pthread.h>
#include <assert.h>
int x;
int nondet_int();
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  x = 1;
  while (nondet_int())
    ;
  x = 0;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x == 0);
  return 0;
}
