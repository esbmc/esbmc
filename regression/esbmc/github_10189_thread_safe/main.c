// #10189: under --sv-comp a program that starts a thread runs incremental BMC
// over its schedules, so the forward condition can prove it.
#include <assert.h>
#include <pthread.h>

int x;

void *set(void *arg)
{
  x = 1;
  return 0;
}

int main()
{
  pthread_t t;
  pthread_create(&t, 0, set, 0);
  for (int i = 0; i < 3; ++i)
    ;
  assert(x == 0 || x == 1);
  return 0;
}
