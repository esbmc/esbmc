/* #10990: the thread's loop is cut by its unwinding bound, which ends the
 * thread with a HALT event instead of an assumption. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  for (int i = 0; i < 10; ++i)
    x = i;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  assert(x == 9);
  return 0;
}
