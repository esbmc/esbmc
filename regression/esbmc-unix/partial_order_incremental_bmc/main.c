/* #10990: incremental BMC raises the bound until the forward condition shows
 * both threads' loops fully unwound. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  for (int i = 0; i < 1; ++i)
  {
    __ESBMC_atomic_begin();
    int r = x;
    x = r + 1;
    __ESBMC_atomic_end();
  }
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
