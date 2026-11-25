// #1991: creating a thread on one branch must not hide the assertions on the
// path where it was never created.
#include <assert.h>
#include <pthread.h>

_Bool nondet_bool(void);

pthread_t c;
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
    return 0;
  }
  d = 3;
  assert(d == 0);
  return 0;
}
