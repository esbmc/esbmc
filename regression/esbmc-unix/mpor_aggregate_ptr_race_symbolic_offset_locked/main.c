#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int *a[2] = {&g, &g};
int **ap;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

// The passing direction of R32: _symbolic_offset with the mutex added, so it
// enters the widest route the walk has (measured, not assumed) and must stay
// SUCCESSFUL. It is a canary, not a witness -- it also passes on unpatched
// code, so it kills no mutant of R32. _widen_contained is the test whose
// success actually depends on the widening staying inside the value set.
void *writer(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&m);
  **ap = 1;
  pthread_mutex_unlock(&m);
  return 0;
}

int main(void)
{
  pthread_t t;
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  ap = &a[i];
  pthread_create(&t, 0, writer, 0);
  pthread_mutex_lock(&m);
  g = 2;
  int seen = g;
  pthread_mutex_unlock(&m);
  pthread_join(t, 0);
  assert(seen == 2);
  return 0;
}
