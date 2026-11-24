/* #10990: a write after the spawn may come after the thread's read. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  assert(x == 1);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  x = 1;
  return 0;
}
