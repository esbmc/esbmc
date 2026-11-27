/* #10990: a compound assignment to a narrow element or pointee assigns to
 * (int)a[i] or (int)*p. It is not atomic, so one increment may be lost, but
 * both may reach main. */
#include <pthread.h>
#include <assert.h>
_Atomic char a[2];
int i = 1;
_Atomic short s;
void *t(void *arg)
{
  _Atomic short *p = &s;
  a[i] += 1;
  *p += 1;
  return 0;
}
int main()
{
  pthread_t x, y;
  pthread_create(&x, 0, t, 0);
  pthread_create(&y, 0, t, 0);
  pthread_join(x, 0);
  pthread_join(y, 0);
  assert(a[1] != 2 || s != 2);
  return 0;
}
