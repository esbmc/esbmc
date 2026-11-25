/* #10990: the read and the write through p are separate events, so the
   other thread's increment may land between them and be lost. */
#include <pthread.h>
#include <assert.h>
int x;
void *t(void *arg)
{
  int *p = &x;
  int v = *p;
  *p = v + 1;
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
