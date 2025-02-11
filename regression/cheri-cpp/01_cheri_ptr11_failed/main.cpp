// TC description:
//  This is a modified version of 01_cheri_ptr11_failed with constructor array initializer

#include <stdint.h>
#include <cheri/cheric.h>

int nondet_int();

class t2
{
public:
  int a[5];

  t2() : a{0,1,2,3,4}
  {
  }

  int bar();
};

void* foo(void *a) {
  return ((void *)((intptr_t)a & (intptr_t)~1));
}

int bar() {
  int a = nondet_int() % 10;
  int *__capability cap_ptr = cheri_ptr(&a, sizeof(a));
  assert(*cap_ptr < 9);
  return 0;
}

int main() {
  t2 t;
  assert(t.a[0] == 0);

  t.bar();

  return 0;
}
