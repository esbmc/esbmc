#include <pthread.h>
#include <assert.h>

/* Two calls to one function stand at the same pc with the same visible state
 * but return to different places. If the state fingerprint omits the return
 * continuation they dedup, and the assertion past the second call is pruned
 * away rather than reported (#6784). */

int shared = 0;

void f(void)
{
  int x = shared;
  (void)x;
}

void *worker(void *p)
{
  f();
  f();
  assert(shared == 0);
  return 0;
}

void *idle(void *p)
{
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, worker, 0);
  pthread_create(&b, 0, idle, 0);
  return 0;
}
