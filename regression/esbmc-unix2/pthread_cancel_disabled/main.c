/* Verify that PTHREAD_CANCEL_DISABLE prevents cancellation at testcancel points:
 * the thread must complete normally and set the flag even when cancelled. */
#include <pthread.h>
#include <assert.h>

int completed = 0;

void *thread_func(void *arg)
{
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
  pthread_testcancel(); /* must NOT cancel: state is DISABLE */
  completed = 1;
  return NULL;
}

int main()
{
  pthread_t t;
  pthread_create(&t, NULL, thread_func, NULL);
  pthread_cancel(t);
  pthread_join(t, NULL);
  assert(completed == 1);
  return 0;
}
