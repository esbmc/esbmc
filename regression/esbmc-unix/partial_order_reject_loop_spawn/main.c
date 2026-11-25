#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  x = 1;
  return 0;
}
void spawn(pthread_t *id)
{
  pthread_create(id, 0, t, 0);
}
int main()
{
  pthread_t id[2];
  for (int i = 0; i < 2; ++i)
    spawn(&id[i]);
  assert(x == 0);
  return 0;
}
