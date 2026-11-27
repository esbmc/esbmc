/* #10990: abort() stops main, and only main: the thread may run before it. */
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
int x;
void *t(void *arg)
{
  assert(x != 1);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  x = 1;
  abort();
  x = 2;
  return 0;
}
