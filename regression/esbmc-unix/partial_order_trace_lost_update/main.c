/* #10990: atomic increments lose no update. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  int tmp = x;
  x = tmp + 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  __ESBMC_atomic_begin();
  int m = x;
  x = m + 1;
  __ESBMC_atomic_end();
  pthread_join(id, 0);
  assert(x == 2);
  return 0;
}
