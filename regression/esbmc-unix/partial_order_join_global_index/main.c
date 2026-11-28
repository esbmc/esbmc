#include <pthread.h>
#include <assert.h>
int i;
int x;
void *f(void *arg)
{
  __ESBMC_atomic_begin();
  x = x + 1;
  __ESBMC_atomic_end();
  return 0;
}
int main(void)
{
  pthread_t t[2];
  for (i = 0; i < 2; i++)
    pthread_create(&t[i], 0, f, 0);
  for (i = 0; i < 2; i++)
    pthread_join(t[i], 0);
  assert(x == 2);
  return 0;
}
