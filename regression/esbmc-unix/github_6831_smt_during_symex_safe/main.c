#include <pthread.h>
#include <assert.h>

int nondet_int();

int g = 0, h = 0;
int *a[2] = {&g, &h};
int **ap;
pthread_mutex_t m;

// Same array-of-pointers select as github_6831_smt_during_symex_crash, but
// the write is ordered, so rebuilding tuple elements after a context pop must
// not lose the facts that make this provable (issue #6831, W3.3).
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
  pthread_mutex_init(&m, 0);
  pthread_create(&t, 0, writer, 0);
  pthread_join(t, 0);
  pthread_mutex_lock(&m);
  assert(g == 0 || g == 1);
  pthread_mutex_unlock(&m);
  return 0;
}
