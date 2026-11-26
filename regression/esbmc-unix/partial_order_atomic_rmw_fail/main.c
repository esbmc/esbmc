/* #10990: both threads' increments of c reach main, so c can be 2. */
#include <pthread.h>
#include <assert.h>
_Atomic char c;
void *t(void *arg)
{
  c += 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(c != 2);
  return 0;
}
