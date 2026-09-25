#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int *a[2] = {&g, &g};
int **ap;

// R32: with a symbolic index the descriptor's offset is unset -- `c:@ap =
// { <a, *, 8, int *[2]> }` -- so there is no offset to spell back out and the
// unrefined lookup of `c:@a` misses the entry holding the answer (`c:@a[]`).
// The walk now takes every path of the dereferenced type when the offset says
// nothing. Well-defined C: the index is assumed in bounds.
void *writer(void *arg)
{
  (void)arg;
  **ap = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  ap = &a[i];
  pthread_create(&t, 0, writer, 0);
  g = 2;
  int seen = g;
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
