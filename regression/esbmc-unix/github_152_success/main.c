#include <assert.h>
#include <pthread.h>
void *t1(void *arg)
{
  //do something
  return NULL;
}
int main()
{
  pthread_t id;
  pthread_create(&id, NULL, t1, NULL);
  return 0;
  assert(0);
}
