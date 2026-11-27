/* #10990: the trace shows both reads of x before either write. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  int tmp = x;
  x = tmp + 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  int m = x;
  x = m + 1;
  pthread_join(id, 0);
  assert(x == 2);
  return 0;
}
