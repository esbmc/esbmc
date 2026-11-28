#include <pthread.h>
#include <stdlib.h>
#include <assert.h>

int cnt = 0, x = 0, y = 0, r0 = 0, r1 = 0;

void assume_abort_if_not(int cond)
{
  if (!cond)
    abort();
}

void *p0(void *arg)
{
  x = 1;
  r0 = y;
  __ESBMC_atomic_begin();
  cnt = cnt + 1;
  __ESBMC_atomic_end();
  return 0;
}

void *p1(void *arg)
{
  y = 1;
  r1 = x;
  __ESBMC_atomic_begin();
  cnt = cnt + 1;
  __ESBMC_atomic_end();
  return 0;
}

int main()
{
  pthread_t t0, t1;
  pthread_create(&t0, 0, p0, 0);
  pthread_create(&t1, 0, p1, 0);
  __ESBMC_atomic_begin();
  int done = cnt == 2;
  __ESBMC_atomic_end();
  assume_abort_if_not(done);
  assert(!(r0 == 1 && r1 == 1));
  return 0;
}
