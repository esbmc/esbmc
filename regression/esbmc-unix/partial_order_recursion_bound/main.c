#include <pthread.h>
#include <assert.h>
int s;
void spin(void)
{
  spin();
}
void *t(void *arg)
{
  __ESBMC_atomic_begin();
  s++;
  assert(s == 1);
  s--;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  spin();
}
