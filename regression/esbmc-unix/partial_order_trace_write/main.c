/* #10990: after the join, main reads t's write. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  int m = x;
  assert(m == 1);
  return 0;
}
