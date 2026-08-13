#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int *a[2] = {&g, &g};
int **ap;

// R32: with a symbolic index the descriptor's offset is unset -- `c:@ap =
// { <a, *, 8, int *[2]> }` -- so R31's walk has no offset to spell back out and
// skips it, leaving the unrefined lookup of `c:@a` to miss the entry that holds
// the answer (`c:@a[]`). A constant index detects the same race; see
// _array_decay. Well-defined C: the index is assumed in bounds.
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
