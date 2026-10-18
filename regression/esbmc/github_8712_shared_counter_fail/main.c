#include <pthread.h>
#include <assert.h>
int g;
void *t(void *arg)
{
  for (g = 0; g < 10; g++)
    ;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  assert(g != 5);
  return 0;
}
