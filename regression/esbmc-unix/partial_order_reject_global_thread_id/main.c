#include <pthread.h>
#include <assert.h>
pthread_t id;
void *t(void *arg)
{
  return 0;
}
int main()
{
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
