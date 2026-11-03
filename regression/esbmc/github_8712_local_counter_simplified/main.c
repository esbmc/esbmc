#include <pthread.h>
#include <assert.h>
int r;
void *t(void *arg)
{
  int i;
  for (i = 0; i < 10; i++)
    ;
  r = i;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  assert(r == 10);
  return 0;
}
