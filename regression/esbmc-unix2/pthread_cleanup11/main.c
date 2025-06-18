#include <pthread.h>
#include <assert.h>
#include <unistd.h>

int flag = 0;

void cleanup(void *arg) { flag = 1; }

void *thread_func(void *arg)
{
  pthread_cleanup_push(cleanup, NULL);
  while (1) { sleep(1); } // infinite loop
  pthread_cleanup_pop(0); // not reached
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  sleep(1); // give time to enter loop
  pthread_cancel(t); // should trigger cleanup
  pthread_join(t, NULL);
  assert(flag == 1); // cleanup must be called
}

