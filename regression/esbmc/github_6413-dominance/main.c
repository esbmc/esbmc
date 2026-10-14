#include <pthread.h>

int x = 0;

void *writer(void *arg)
{
  x = 1;
  return NULL;
}

int main()
{
  pthread_t thrd;
  pthread_create(&thrd, NULL, writer, NULL);
  // Holds in the interleaving where main runs first, is violated in the one
  // where the writer runs first. The run must report it once, as FAILED.
  __ESBMC_assert(x == 0, "x is still zero");
  pthread_join(thrd, NULL);
  return 0;
}
