// #1991: the thread created on one branch is not visible on the path where it
// was never created.
#include <assert.h>
#include <pthread.h>

_Bool nondet_bool(void);

pthread_t c;
int created;
int d;

void *e(void *arg)
{
  d = 6;
  return 0;
}

int main()
{
  if (nondet_bool())
  {
    pthread_create(&c, 0, e, 0);
    created = 1;
    return 0;
  }
  assert(created == 0);
  assert(d == 0);
  return 0;
}
