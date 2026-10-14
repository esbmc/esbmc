#include <pthread.h>
#include <assert.h>

int g = 0;

struct Inner
{
  int *p;
};

struct Outer
{
  struct Inner in;
};

struct Outer o = {{&g}};

// R29, nested: the flat case needs one member peeled off the constant struct,
// this one needs two. Descending a single level would leave `.p` unconsumed
// against the inner constant_struct2t and lose the object again.
void *writer(void *arg)
{
  (void)arg;
  *(o.in.p) = 1;
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
