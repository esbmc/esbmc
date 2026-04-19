#include <pthread.h>
#include <assert.h>

int flags[3] = {0, 0, 0};

void cleanup0(void *arg) { flags[0] = 1; }
void cleanup1(void *arg) { flags[1] = 1; }
void cleanup2(void *arg) { flags[2] = 1; }

void *thread_func(void *arg)
{
  pthread_cleanup_push(cleanup0, NULL);
  pthread_cleanup_push(cleanup1, NULL);
  pthread_cleanup_push(cleanup2, NULL);
  pthread_cleanup_pop(1); // cleanup2 run
  pthread_cleanup_pop(1); // cleanup1 run
  pthread_cleanup_pop(1); // cleanup0 run
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(flags[0] == 1);
  assert(flags[1] == 1);
  assert(flags[2] == 0); // should fail
}

