/* #9441: a compound assignment to an _Atomic object narrower than int is one
 * read-modify-write (C11 6.5.16.2p3), so neither thread loses an update. */
#include <assert.h>
#include <pthread.h>

_Atomic char c = 0;
_Atomic short s = 0;
_Atomic _Bool b = 0;
_Atomic unsigned char u = 1;

void *t(void *arg)
{
  c += 1;
  s -= 1;
  b ^= 1;
  u <<= 1;
  return 0;
}

int main(void)
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, t, 0);
  pthread_create(&t2, 0, t, 0);
  pthread_join(t1, 0);
  pthread_join(t2, 0);
  assert(c == 2 && s == -2 && b == 0 && u == 4);
  return 0;
}
