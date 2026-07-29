/* call_once must invoke its function at most once across all threads. */
#include <threads.h>
#include <assert.h>

static once_flag flag = ONCE_FLAG_INIT;
static int call_count = 0;

static void initializer(void)
{
  call_count++;
}

static int worker(void *arg)
{
  (void)arg;
  call_once(&flag, initializer);
  return 0;
}

int main(void)
{
  thrd_t t;
  thrd_create(&t, worker, NULL);
  call_once(&flag, initializer);
  thrd_join(t, NULL);
  assert(call_count == 1);
  return 0;
}
