/* #10990: a thread halted by an unwinding bound inside an atomic section
 * still made its write before the section visible. */
#include <pthread.h>
#include <assert.h>
int x;
int nondet_int();
void *t(void *arg)
{
  x = 1;
  __ESBMC_atomic_begin();
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
