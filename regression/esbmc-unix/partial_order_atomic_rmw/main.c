/* #10990: a compound assignment to an _Atomic object narrower than int
 * assigns to (int)c, and is still recorded as a write of c. */
#include <pthread.h>
#include <assert.h>
_Atomic char c;
_Atomic short s;
void *t(void *arg)
{
  c += 1;
  s -= 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(c == 2 && s == -2);
  return 0;
}
