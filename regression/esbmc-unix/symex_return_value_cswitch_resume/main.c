#include <pthread.h>
#include <assert.h>

int x = 0, z = 0;

int notify(void)
{
  return 1;
}

/* The observer can only set z at the return boundary of notify(), and the
   writer only reads z after resuming from it, so the assertion fires only if
   a switch is offered there *and* the writer survives it. */
void *writer(void *arg)
{
  x = notify();
  x = 2;
  assert(z != 7);
  return 0;
}

void *observer(void *arg)
{
  if (x == 1)
    z = 7;
  return 0;
}

int main(void)
{
  pthread_t w, o;
  pthread_create(&w, 0, writer, 0);
  pthread_create(&o, 0, observer, 0);
  return 0;
}
