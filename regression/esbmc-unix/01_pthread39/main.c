#include <pthread.h>
#include <assert.h>

int z;

void *t1(void *arg)
{
  z=3;
}

void *t2(void *arg)
{
  assert(z==0);
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);

  return 0;
}
