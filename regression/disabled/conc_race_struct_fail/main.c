/* conc_race_struct_fail:
 * Two threads update different fields of a shared struct without synchronisation.
 * Thread 1 writes s.x; Thread 2 writes s.y.
 * A field-level data race on s.x occurs when both threads also read s.x.
 *
 * Expected: VERIFICATION FAILED (data race on struct field s.x or s.y)
 */
#include <pthread.h>

typedef struct { int x; int y; int sum; } Stats;
Stats s = {0, 0, 0};

void *worker1(void *arg)
{
  s.x = 10;           /* write s.x */
  s.sum = s.x + s.y;  /* read both — races with worker2 writing s.y */
  return NULL;
}

void *worker2(void *arg)
{
  s.y = 20;           /* write s.y */
  s.sum = s.x + s.y;  /* read both — races with worker1 */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, worker1, NULL);
  pthread_create(&t2, NULL, worker2, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
