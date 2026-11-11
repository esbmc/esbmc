// #10293: q is loaded from uninitialised memory, so it hits no object symex
// models and cannot race with main's write to g.
#include <pthread.h>
#include <stdlib.h>

struct S
{
  int *p;
};

int g;

void *t1(void *arg)
{
  struct S *s = malloc(sizeof(struct S));
  __ESBMC_assume(s);
  int *q = s->p;
  if (q)
    *q = *q + 1;
  return 0;
}

int main()
{
  pthread_t id;
  pthread_create(&id, 0, t1, 0);
  g = 1;
  return 0;
}
