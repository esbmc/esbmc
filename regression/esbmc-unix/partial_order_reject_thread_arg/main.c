#include <pthread.h>
#include <assert.h>
void *t(void *arg)
{
  *(int *)arg = 1;
  return 0;
}
int main()
{
  int v = 0;
  pthread_t id;
  pthread_create(&id, 0, t, &v);
  assert(v == 0);
  return 0;
}
