#include <assert.h>
#include <pthread.h>

/* The violation needs several context switches: `hits` only reaches 3 when the
 * two threads fully alternate. Unbounded DFS can dive past such a schedule;
 * deepening visits schedules in order of switch count and reaches it. */

static int turn = 0, hits = 0;

static void *A(void *x)
{
  for (int i = 0; i < 3; i++)
    if (turn == 0)
    {
      turn = 1;
      hits++;
    }
  return 0;
}

static void *B(void *x)
{
  for (int i = 0; i < 3; i++)
    if (turn == 1)
      turn = 0;
  return 0;
}

int main(void)
{
  pthread_t p, q;

  pthread_create(&p, 0, A, 0);
  pthread_create(&q, 0, B, 0);
  pthread_join(p, 0);
  pthread_join(q, 0);
  assert(hits < 3);

  return 0;
}
