/* The worker reads main's local only through memchr, which symex runs as an
 * intrinsic: MPOR must still key the read, or the schedule where main's
 * write lands before it is pruned (#7826). */
#include <assert.h>
#include <pthread.h>
#include <string.h>

int g;

void *worker(void *arg)
{
  if (memchr(arg, 5, sizeof(int)))
    g = 1;
  return 0;
}

int main()
{
  int done = 0;
  pthread_t t;
  pthread_create(&t, 0, worker, &done);
  done = 5;
  done = 0;
  assert(g == 0);
  return 0;
}
