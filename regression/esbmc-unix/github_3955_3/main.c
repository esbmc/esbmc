/* counter = counter + 1 on an _Atomic int splits into two separate atomic
 * operations (load and store) with a context-switch window between them.
 * Both threads reading 0, then both storing 1 (lost update) is therefore
 * a reachable execution, making counter == 2 not always true.
 * Exercises the both-sides-atomic path with self-dependency. */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

atomic_int counter = 0;

void *increment(void *arg)
{
  counter = counter + 1; /* atomic load then atomic store — two separate ops */
  return NULL;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, NULL, increment, NULL);
  pthread_create(&t2, NULL, increment, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  assert(counter == 2); /* can fail: lost update gives counter == 1 */
  return 0;
}
