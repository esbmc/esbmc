/* #10990: a join on one of two threads leaves the other unfinished. */
#include <pthread.h>
#include <assert.h>
int nondet_int();
int a, b;
void *ta(void *arg)
{
  a = 1;
  return 0;
}
void *tb(void *arg)
{
  b = 1;
  return 0;
}
int main()
{
  pthread_t id[2];
  pthread_create(&id[0], 0, ta, 0);
  pthread_create(&id[1], 0, tb, 0);
  int j = nondet_int() ? 1 : 0;
  pthread_join(id[j], 0);
  assert(a == 1 && b == 1);
  return 0;
}
