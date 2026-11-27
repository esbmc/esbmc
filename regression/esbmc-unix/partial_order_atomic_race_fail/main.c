/* #10990, #2386: a write inside an atomic section races with a plain read. */
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
  int l = g;
  pthread_join(t, 0);
  return 0;
}
