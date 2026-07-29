#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

/* Negative counterpart of esbmc-unix/github_2174. The read-modify-write is
 * split into a non-atomic load and store, so one update can be lost and the
 * counter may end at 1. This pins that the atomic model above is doing real
 * work rather than the interleaving simply never being explored. */
atomic_int counter;

void *worker(void *arg)
{
  int tmp = atomic_load(&counter);
  atomic_store(&counter, tmp + 1);
  return 0;
}

int main(void)
{
  pthread_t p, q;

  atomic_init(&counter, 0);
  pthread_create(&p, 0, worker, 0);
  pthread_create(&q, 0, worker, 0);
  pthread_join(p, 0);
  pthread_join(q, 0);

  assert(atomic_load(&counter) == 2);
  return 0;
}
