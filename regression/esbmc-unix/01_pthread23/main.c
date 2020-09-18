#include <pthread.h>

int g;

void *t1(void *arg)
{
  g=2;
}

int main()
{
  pthread_t id1, id2;
  g=1;

  assert(g==1);
  pthread_create(&id1, NULL, t1, NULL);
}
