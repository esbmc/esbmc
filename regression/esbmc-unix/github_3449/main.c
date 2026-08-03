#include <assert.h>
#include <threads.h>

/* C11 threads layered on the pthread model: thrd_join must deliver the entry
 * point's int result, and a mtx_t must serialise the increment. See #3449. */

static mtx_t m;
static int counter = 0;

static int worker(void *arg)
{
  mtx_lock(&m);
  counter++;
  mtx_unlock(&m);
  return *(int *)arg + 1;
}

int main(void)
{
  thrd_t a, b;
  int seed = 41, res = 0;

  mtx_init(&m, mtx_plain);
  assert(thrd_create(&a, worker, &seed) == thrd_success);
  assert(thrd_create(&b, worker, &seed) == thrd_success);
  assert(thrd_join(a, &res) == thrd_success);
  assert(res == 42);
  assert(thrd_join(b, 0) == thrd_success);
  assert(counter == 2);
  mtx_destroy(&m);

  return 0;
}
