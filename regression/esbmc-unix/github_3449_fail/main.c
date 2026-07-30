#include <assert.h>
#include <threads.h>

/* Negative counterpart of esbmc-unix/github_3449: without the mutex the
 * read-modify-write can lose an update, so the counter may end at 1. Pins that
 * the passing test above really is exploring the interleavings. */

static int counter = 0;

static int worker(void *arg)
{
  int tmp = counter;
  counter = tmp + 1;
  return 0;
}

int main(void)
{
  thrd_t a, b;

  thrd_create(&a, worker, 0);
  thrd_create(&b, worker, 0);
  thrd_join(a, 0);
  thrd_join(b, 0);
  assert(counter == 2);

  return 0;
}
