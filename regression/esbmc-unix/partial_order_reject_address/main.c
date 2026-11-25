#include <pthread.h>
#include <assert.h>
int x;
void touch(int *p);
void *t(void *arg)
{
  touch(&x);
  return 0;
}
int main()
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(x == 0);
  return 0;
}
