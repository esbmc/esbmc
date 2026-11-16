// #9867: a plain member next to an _Atomic one still races.
#include <pthread.h>
struct S { _Atomic int f; int g; };
struct S s;

void *thr(void *arg)
{
  s.f = 1;
  s.g = 1;
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, thr, 0);
  s.f = 2;
  s.g = 2;
  pthread_join(id, 0);
  return 0;
}
