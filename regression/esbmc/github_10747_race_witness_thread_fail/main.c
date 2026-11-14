#include <pthread.h>
int g;
void *t(void *arg)
{
  g = 1;
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  g = 2;
  pthread_join(id, 0);
  return 0;
}
