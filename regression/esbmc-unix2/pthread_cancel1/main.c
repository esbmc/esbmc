#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

int reached = 0;

void *thread_func(void *arg)
{
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
  pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

  while (1)
  {
    pthread_testcancel(); // cancellation point
    sleep(1);
  }

  reached = 1; // should not happen
  return NULL;
}

int main()
{
  pthread_t tid;
  pthread_create(&tid, NULL, thread_func, NULL);

  sleep(2);
  pthread_cancel(tid);

  void *res;
  pthread_join(tid, &res);

  // Assert thread was canceled and did not complete normally
  assert(reached == 0);

  return 0;
}

