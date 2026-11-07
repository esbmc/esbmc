// #9906: the jmp_buf was set by a frame of the thread that has returned
#include <assert.h>
#include <pthread.h>
#include <setjmp.h>
jmp_buf b;
void f(void)
{
  setjmp(b);
}
void *t(void *arg)
{
  f();
  longjmp(b, 1);
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
