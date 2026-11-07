// #9906: the thread's own frame still holds the active setjmp
#include <assert.h>
#include <pthread.h>
#include <setjmp.h>
_Thread_local jmp_buf b;
void f(void)
{
  longjmp(b, 3);
}
void *t(void *arg)
{
  int r = setjmp(b);
  if (r == 0)
    f();
  assert(r == 3);
  return 0;
}
int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, t, 0);
  pthread_join(id, 0);
  return 0;
}
