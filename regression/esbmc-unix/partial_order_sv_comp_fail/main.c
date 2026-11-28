/* #10990: --sv-comp gives a threaded program falsification rounds, which
 * --partial-order rejects; it must run without them and find the lost update. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  int r = x;
  x = r + 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(x == 2);
  return 0;
}
