// #9867: an _Atomic member or element is atomic even when its container is not.
#include <pthread.h>
struct I { _Atomic int f; };
struct S { _Atomic int f; struct I i; };
union U { _Atomic int f; long l; };
struct S s;
struct S *ps = &s;
union U u;
_Atomic int a[2][2];
_Atomic int w[4];
_Atomic int *pv = &w[2];

void write_all(int v)
{
  s.f = v;
  s.i.f = v;
  ps->f = v;
  u.f = v;
  a[1][1] = v;
  *(pv - 1) = v;
}

void *thr(void *arg)
{
  write_all(1);
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, thr, 0);
  write_all(2);
  pthread_join(id, 0);
  return 0;
}
