/* #10990: the trace shows t's write before main reads it. */
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
  int m = x;
  assert(m == 0);
  return 0;
}
