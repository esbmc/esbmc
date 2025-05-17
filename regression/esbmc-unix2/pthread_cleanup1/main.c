#include <pthread.h>
#include <assert.h>

int flag = 0;

void cleanup(void *arg)
{
  flag = *(int *)arg;
}

void *thread_func(void *arg)
{
  int val = 42;
  pthread_cleanup_push(cleanup, &val);
  pthread_cleanup_pop(1); // should call cleanup
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(flag == 42); // must be true
}


