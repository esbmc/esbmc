/* #10990: after a join, main sees the joined thread's last write. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  x = 1;
  x = 2;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  assert(x == 2);
  return 0;
}
