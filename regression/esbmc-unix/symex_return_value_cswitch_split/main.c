#include <pthread.h>
#include <assert.h>

int x = 0;

int notify(void)
{
  return 1;
}

void *writer(void *arg)
{
  /* Same program as symex_return_value_cswitch, except the return value lands
     in a local first, so the shared write is a plain ASSIGN. */
  int v = notify();
  x = v;
  x = 2;
  return 0;
}

void *observer(void *arg)
{
  assert(x != 1);
  return 0;
}

int main(void)
{
  pthread_t w, o;
  pthread_create(&w, 0, writer, 0);
  pthread_create(&o, 0, observer, 0);
  return 0;
}
