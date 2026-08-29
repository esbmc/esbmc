#include <assert.h>
#include <pthread.h>

/* --incremental-context-bound must still terminate with a real proof, not a
 * bounded one: it deepens until a round completes without the bound cutting
 * any interleaving, and only then reports success. See issue #6480. */

static pthread_mutex_t m;
static int c = 0;

static void *worker(void *arg)
{
  pthread_mutex_lock(&m);
  c++;
  pthread_mutex_unlock(&m);
  return 0;
}

int main(void)
{
  pthread_t p, q;

  pthread_mutex_init(&m, 0);
  pthread_create(&p, 0, worker, 0);
  pthread_create(&q, 0, worker, 0);
  pthread_join(p, 0);
  pthread_join(q, 0);
  assert(c == 2);

  return 0;
}
