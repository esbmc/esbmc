#include <assert.h>
#include <pthread.h>

int g = 0, h = 0;
int *a[2] = {&h, &g};
int **ap;

// The writer's target is g or h. Keying the write by only the first shared
// object it may reach made MPOR prune the interleaving that races on g.
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
