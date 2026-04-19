#include <assert.h>
#include <pthread.h>

void *thread1(void *arg)
{
  return 0;
}
void *thread0(void *arg)
{
  pthread_t t2;
  pthread_create(&t2, 0, thread1, 0);
  pthread_join(t2, 0);
  return 0;
}
int main(void)
{
  pthread_t t;
  pthread_create(&t, 0, thread0, 0);
  pthread_join(t, 0);
  assert(0);
  return 0;
}
