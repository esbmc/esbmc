#include <pthread.h>
#include <assert.h>

int g = 0;

struct Outer
{
  struct
  {
    int *p;
  };
};

struct Outer o = {{&g}};

// R29, anonymous member: clang names this member
// "struct Outer::(anonymous at main.c:8:3)", whose text contains '.', so
// splitting the value-set suffix on the next '.' or '[' cut the name apart and
// found no component. The leading component is matched against the declared
// names instead.
void *writer(void *arg)
{
  (void)arg;
  *(o.p) = 1;
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
