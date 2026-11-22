// #10189: pthread_create in a function main never calls, as in Juliet's
// support code, leaves the program sequential, so k-induction proves it.
#include <assert.h>
#include <pthread.h>

void *work(void *arg)
{
  return 0;
}

void start_thread()
{
  pthread_t t;
  pthread_create(&t, 0, work, 0);
}

int main()
{
  unsigned i = 0;
  while (i < 10)
    ++i;
  assert(i == 10);
  return 0;
}
