#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

/* Negative counterpart of esbmc-unix/github_2174: splitting the read-modify-
 * write can lose an update, so the atomicity there is not a missed
 * interleaving. */
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
