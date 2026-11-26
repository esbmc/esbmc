/* #10990: each unwinding of a spawning loop creates its own thread. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  x = x + 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t id[3];
  for (int i = 0; i < 3; ++i)
    pthread_create(&id[i], 0, t, 0);
  for (int i = 0; i < 3; ++i)
    pthread_join(id[i], 0);
  assert(x == 3);
  return 0;
}
