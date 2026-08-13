#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0;
int *a[2] = {&g, &g};
int **ap;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

// The passing direction of R32. An unset offset selects no path, so the descent
// takes every path of the dereferenced type -- the widest thing this walk ever
// does. This test fails if that width ever turns a correct program into a false
// alarm; _symbolic_offset is the same shape with the mutex removed.
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
