#include <pthread.h>
#include <assert.h>

int counter = 0;

void inc(void *arg) { counter++; }

void *thread_func(void *arg)
{
  pthread_cleanup_push(inc, NULL);
  pthread_cleanup_push(inc, NULL);
  pthread_cleanup_pop(0);
  pthread_cleanup_pop(0);
  pthread_exit(NULL);
}

int main(void)
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(counter == 2); // both handlers ran
  return 0;
}

