/* #10990: an element increment inside an atomic section is not lost. */
#include <pthread.h>
#include <assert.h>
int a[2];
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  int r = a[1];
  a[1] = r + 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t, 0);
  pthread_create(&y, 0, t, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  assert(a[1] == 2);
  return 0;
}
