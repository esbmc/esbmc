#include <pthread.h>
#include <assert.h>

int counter = 0;

void inc(void *arg) { counter++; }

void *thread_func(void *arg)
{
  pthread_cleanup_push(inc, NULL);
  pthread_cleanup_push(inc, NULL);
  pthread_cleanup_pop(1);
  pthread_exit(NULL);
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(counter == 2);
}

