/* VERIFICATION FAILED: asserts cleanup never runs, but since the thread
 * cancels itself before testcancel, cleanup always executes. */
#include <pthread.h>
#include <assert.h>

int flag = 0;

void cleanup(void *arg)
{
  flag = 1;
}

void *thread_func(void *arg)
{
  pthread_cleanup_push(cleanup, NULL);
  pthread_cancel(pthread_self());
  pthread_testcancel();
  pthread_cleanup_pop(0);
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_join(t, NULL);
  assert(flag == 0); /* must fail: cleanup always runs on self-cancel */
  return 0;
}
