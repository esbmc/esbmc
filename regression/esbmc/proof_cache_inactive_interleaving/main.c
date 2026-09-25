#include <assert.h>
#include <pthread.h>

int x;

void *inc(void *arg)
{
  x = x + 1;
  return 0;
}

int main(void)
{
  x = nondet_int();
  __ESBMC_assume(x > 0 && x < 10);

  pthread_t t;
  pthread_create(&t, 0, inc, 0);
  x = x + 1;
  pthread_join(t, 0);
  assert(x > 0);
  return 0;
}
