#include <pthread.h>
#include <assert.h>
#include <stdlib.h>

int flag = 0;

void cleanup(void *arg)
{
  free(arg);
  flag = 1;
}

void *thread_func(void *arg)
{
  void *mem = malloc(16);
  pthread_cleanup_push(cleanup, mem);
  pthread_cleanup_pop(0); // triggers free
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(flag == 1);
}

