/* The worker reads main's local only through memmove, which symex runs as an
 * intrinsic: MPOR must still key the read, or the schedule where main's
 * write lands before it is pruned (#7826). */
#include <assert.h>
#include <pthread.h>
#include <string.h>

int g;

void *worker(void *arg)
{
  char x[2];
  memmove(x, arg, 2);
  if (x[0] != 0)
    g = 1;
  return 0;
}

int main()
{
  char done[2] = {0, 0};
  pthread_t t;
  pthread_create(&t, 0, worker, done);
  done[0] = 5;
  done[0] = 0;
  assert(g == 0);
  return 0;
}
