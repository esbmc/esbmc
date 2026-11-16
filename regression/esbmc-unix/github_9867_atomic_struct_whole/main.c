// #9867: an _Atomic struct, or an array of them, assigned as a whole.
#include <pthread.h>
struct S { int f; };
typedef _Atomic struct S at_s;
_Atomic struct S s;
struct S _Atomic e;
at_s t;
_Atomic struct S arr[4];
struct S other;

void write_all(void)
{
  s = other;
  e = other;
  t = other;
  arr[1] = other;
}

void *thr(void *arg)
{
  write_all();
  return 0;
}

int main(void)
{
  pthread_t id;
  pthread_create(&id, 0, thr, 0);
  write_all();
  pthread_join(id, 0);
  return 0;
}
