/* #10990: a read observes the latest write before it, not an overwritten one. */
#include <pthread.h>
#include <assert.h>
int x;
void *t1(void *arg)
{
  x = 1;
  x = 2;
  int r = x;
  assert(r != 1);
  return 0;
}
int main()
{
  pthread_t a;
  pthread_create(&a, 0, t1, 0);
  return 0;
}
