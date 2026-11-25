/* #10990: a thread's reads follow the writes its creator made before
   creating it. */
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
  x = 1;
  pthread_create(&id, 0, t, 0);
  return 0;
}
