/* C11 threads without synchronization: ESBMC must catch the data race
 * making the post-join value of `shared` non-deterministic. */
#include <threads.h>
#include <assert.h>

static int shared = 0;

static int worker(void *arg)
{
  (void)arg;
  shared = 5;
  return 0;
}

int main(void)
{
  thrd_t t;
  thrd_create(&t, worker, NULL);
  thrd_join(t, NULL);
  /* Deliberately wrong: writer set shared=5, but assertion expects 0. */
  assert(shared == 0);
  return 0;
}
