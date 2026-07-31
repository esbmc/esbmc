#include <assert.h>
#include <pthread.h>

_Bool receive = 0;
int n = 0;
int dummy = 0;

void *t1(void *arg)
{
  if (receive)
    receive = 0;
  assert(!receive || n);
  return NULL;
}

int main()
{
  pthread_t id;
  n = nondet_int() != 0;
  pthread_create(&id, NULL, t1, NULL);
  if (n)
    receive = 1;
  if (dummy)
    dummy = 0;
  return 0;
}
