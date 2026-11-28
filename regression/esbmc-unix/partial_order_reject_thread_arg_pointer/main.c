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
  int *p = &v;
  pthread_t id;
  pthread_create(&id, 0, t, p);
  pthread_join(id, 0);
  assert(v == 1);
  return 0;
}
