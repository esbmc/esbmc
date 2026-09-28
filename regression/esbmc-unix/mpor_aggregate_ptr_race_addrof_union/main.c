#include <pthread.h>
#include <assert.h>

int g = 0;

union U
{
  long l;
  int *p;
};

union U u = {.p = &g};
int **pp = &u.p;

// R31, union member: every member of a union starts at the same byte, so the
// offset alone cannot say which one is live and the descent has to consider
// each -- only those whose type matches the dereference contribute a path. The
// pointer is the second member so that laying the members out end to end, as a
// struct's are, would miss it.
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
