#include <pthread.h>

// The `g == 1` direction is reachable only when `worker` runs before main's
// test, i.e. under a thread interleaving other than the first one explored.
// Reporting the advisory per interleaving flagged it as provably-dead CWE-561
// on the strength of the first interleaving alone (issue #4495).
int g = 0;

void *worker(void *arg)
{
  g = 1;
  return 0;
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  if (g == 1)
    return 0;
  return 1;
}
