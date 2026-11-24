#include <pthread.h>
#include <assert.h>
int a[2];
void *t(void *arg)
{
  a[0] = 1;
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(a[0] == 0);
  return 0;
}
