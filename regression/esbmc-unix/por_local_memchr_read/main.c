/* The worker reads main's local only through memchr, which symex runs as an
 * intrinsic: MPOR must still key the read, or the schedule where main's
 * write lands before it is pruned (#7826). */
#include <assert.h>
#include <pthread.h>
#include <string.h>

void *worker(void *arg)
{
  const char *p = memchr(arg, 5, 1);
  assert(p == 0 || p == arg);
  return 0;
}

int main()
{
  int done = 0;
  pthread_t t;
  pthread_create(&t, 0, worker, &done);
  done = 5;
  done = 0;
  return 0;
}
