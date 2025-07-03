#include <pthread.h>
#include <assert.h>

int log_file[2] = {0, 0};

void cleanup1(void *arg) { log_file[0] = 1; }
void cleanup2(void *arg) { log_file[1] = 1; }

void *thread_func(void *arg)
{
  pthread_cleanup_push(cleanup1, NULL);
  pthread_cleanup_push(cleanup2, NULL);
  pthread_cleanup_pop(1); // executes cleanup2
  pthread_cleanup_pop(0); // does NOT execute cleanup1
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(log_file[0] == 0); // cleanup1 not executed
  assert(log_file[1] == 1); // cleanup2 executed
}
