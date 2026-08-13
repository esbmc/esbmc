#include <pthread.h>
#include <assert.h>

int g = 0;

struct S
{
  int *p;
};

struct S s = {&g};
int **pp = &s.p;

// R31: taking the address of a struct member and dereferencing twice through it
// prunes the race, exactly as R29 did one level in. This is ordinary C, not
// punning -- &s.p is a well-defined int**. Reports SUCCESSFUL by default and
// FAILED under --no-por, both solvers; copying the pointer to a local first
// (see _addrof_local) restores detection, so the boundary is syntactic.
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
