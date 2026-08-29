#include <pthread.h>

/* Two threads whose accesses to x and y conflict in both directions, so the
 * reduction has something to prune and the counters have something to report.
 * main returns without joining, so schedules continue past its thread ending. */

static int x, y;

static void *ta(void *arg)
{
  x = 1;
  y = x + 1;
  return 0;
}

static void *tb(void *arg)
{
  y = 2;
  x = y + 1;
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, ta, 0);
  pthread_create(&b, 0, tb, 0);
  return 0;
}
