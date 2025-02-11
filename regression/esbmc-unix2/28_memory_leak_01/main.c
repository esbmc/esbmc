#include <stdlib.h>
#include <pthread.h>

int *shared;

void *t1(void *arg)
{
  return NULL;
}

int main()
{
  shared = (int *)malloc(sizeof(int));
  pthread_t id1;
  pthread_join(id1, NULL);
  free(shared);
  return 0;
}
