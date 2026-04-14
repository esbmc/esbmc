/* atomicity09: two threads concurrently read and modify a shared global.
   Thread 1: x = x + 1  (non-atomic read-modify-write)
   Thread 2: x = x + 2
   The atomicity checker should detect that x may be modified between
   the snapshot and the assignment, yielding VERIFICATION FAILED. */
#include <pthread.h>

int x = 0;

void *t1(void *arg)
{
  x = x + 1;
  return NULL;
}

void *t2(void *arg)
{
  x = x + 2;
  return NULL;
}

int main(void)
{
  pthread_t id1, id2;
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  return 0;
}
