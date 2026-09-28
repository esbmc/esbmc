#include <assert.h>
#include <threads.h>

/* thrd_exit's code, including its sign, must reach thrd_join. A model that
 * dropped the exit value would leave `res` unconstrained and could satisfy
 * this wrong expectation. */

static int worker(void *arg)
{
  thrd_exit(-5);
}

int main(void)
{
  thrd_t t;
  int res = 0;

  thrd_create(&t, worker, 0);
  thrd_join(t, &res);
  assert(res == 5); /* the real value is -5 */

  return 0;
}
