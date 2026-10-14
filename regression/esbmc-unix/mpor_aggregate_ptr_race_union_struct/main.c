#include <pthread.h>
#include <assert.h>

int g = 0;

struct Inner
{
  int *p;
};

union U
{
  struct Inner in;
  long l;
};

union U u = {{&g}};

// R29, struct inside a union: the constant-union case passed its suffix on
// unconsumed, so ".in.p" was looked up in struct Inner, which has no member
// named "in". The initialised member's component is consumed first.
void *writer(void *arg)
{
  (void)arg;
  *(u.in.p) = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
