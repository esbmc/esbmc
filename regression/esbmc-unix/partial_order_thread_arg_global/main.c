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
  pthread_join(id, 0);
  assert(g == 1);
  return 0;
}
