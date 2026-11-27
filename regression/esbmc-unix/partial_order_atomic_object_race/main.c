/* #10990: accesses to an _Atomic object do not race (C11 5.1.2.4p25). */
#include <pthread.h>
#include <stdatomic.h>
atomic_uint flag;
_Atomic char c;
void *writer(void *arg)
{
  flag = 1;
  c += 1;
  return 0;
}
int main()
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  unsigned v = flag;
  c += 2;
  pthread_join(t, 0);
  return 0;
}
