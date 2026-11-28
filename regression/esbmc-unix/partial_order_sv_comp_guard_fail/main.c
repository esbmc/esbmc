/* Peterson's lock without the turn variable lets both threads in: the
 * --partial-order run under --sv-comp still reports it. */
#include <pthread.h>
#include <assert.h>
int flag1, flag2, x;
void *t1(void *arg)
{
  flag1 = 1;
  if (flag2 == 1)
    return 0;
  x = 0;
  if (x > 0)
    assert(0);
  flag1 = 0;
  return 0;
}
void *t2(void *arg)
{
  x = 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
