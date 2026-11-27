/* #10990: the thread's loop is cut by its unwinding bound, which does not
 * stop main from reading x before its join. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  for (int i = 0; i < 10; ++i)
    x = i + 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x != 2);
  pthread_join(id, 0);
  return 0;
}
