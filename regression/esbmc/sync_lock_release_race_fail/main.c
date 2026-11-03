#include <pthread.h>
#include <assert.h>
int lock, n;
void *t(void *a)
{
  __sync_lock_release(&lock);
  int v = n;
  n = v + 1;
  return 0;
}
int main()
{
  pthread_t a, b;
  pthread_create(&a, 0, t, 0);
  pthread_create(&b, 0, t, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  assert(n == 2);
}
