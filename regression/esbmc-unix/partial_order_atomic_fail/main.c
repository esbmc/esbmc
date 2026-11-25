/* #10990: an atomic section excludes other threads only while it runs, so
   the other thread may increment between the read and the write here. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  int r = x;
  __ESBMC_atomic_end();
  __ESBMC_atomic_begin();
  x = r + 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
