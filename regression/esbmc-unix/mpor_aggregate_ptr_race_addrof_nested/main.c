#include <pthread.h>
#include <assert.h>

int g = 0;

struct I
{
  int *p;
};

struct O
{
  struct I in;
};

struct O o = {{&g}};
int **pp = &o.in.p;

// R31, two levels down: one byte offset has to spell out a two-component path,
// so the descent recurses rather than matching a single member.
void *writer(void *arg)
{
  (void)arg;
  **pp = 1;
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
