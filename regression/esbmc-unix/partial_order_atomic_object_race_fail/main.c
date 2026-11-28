/* #10990: an _Atomic object does not hide a race on a plain one. */
#include <pthread.h>
#include <stdatomic.h>
atomic_uint flag;
unsigned g;
void *writer(void *arg)
{
  flag = 1;
  g = 1;
  return 0;
}
int main()
{
  pthread_t t;
  pthread_create(&t, 0, writer, 0);
  unsigned v = flag;
  unsigned w = g;
  pthread_join(t, 0);
  return 0;
}
