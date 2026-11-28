/* #10990: a thread writes through its argument, the address of a global. */
#include <pthread.h>
#include <assert.h>
int g;
void *t(void *arg)
{
  *(int *)arg = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, &g);
  assert(g == 0);
  return 0;
}
