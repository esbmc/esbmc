#include <pthread.h>
#include <assert.h>
int a[2];
void *t(void *arg)
{
  int *p = a;
  p[1] = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(a[1] == 0);
  return 0;
}
