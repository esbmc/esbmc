/* #10990: an increment split over two really-atomic sections can be lost. */
#include <pthread.h>
#include <assert.h>
void __ESBMC_really_atomic_begin(void);
void __ESBMC_really_atomic_end(void);
int x;
void *t(void *arg)
{
  __ESBMC_really_atomic_begin();
  int tmp = x;
  __ESBMC_really_atomic_end();
  __ESBMC_really_atomic_begin();
  x = tmp + 1;
  __ESBMC_really_atomic_end();
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
