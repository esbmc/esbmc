#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

/* Two concurrent atomic read-modify-writes cannot lose an update, so the
 * counter is exactly 2. See #2174. */
atomic_int counter;

void *worker(void *arg)
{
  atomic_fetch_add(&counter, 1);
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
