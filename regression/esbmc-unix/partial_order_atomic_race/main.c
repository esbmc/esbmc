/* #10990, #2386: accesses inside atomic sections do not race. */
#include <pthread.h>
int g, h;
void *writer(void *arg)
{
  __ESBMC_atomic_begin();
  g = 1;
  h = 1;
  __ESBMC_atomic_end();
  return 0;
}
int main()
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  __ESBMC_atomic_begin();
  int l = g;
  __ESBMC_atomic_end();
  pthread_join(t, 0);
  return 0;
}
