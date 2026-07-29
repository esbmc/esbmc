/* C11 thread create/join with mutex protects shared counter. */
#include <threads.h>
#include <assert.h>

static int shared = 0;
static mtx_t m;

static int worker(void *arg)
{
  (void)arg;
  mtx_lock(&m);
  shared++;
  mtx_unlock(&m);
  return 0;
}

int main(void)
{
  thrd_t t;
  mtx_init(&m, mtx_plain);
  thrd_create(&t, worker, NULL);
  thrd_join(t, NULL);
  mtx_destroy(&m);
  assert(shared == 1);
  return 0;
}
