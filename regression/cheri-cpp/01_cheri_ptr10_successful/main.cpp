// TC description:
//  This is a modified version of 01_cheri_ptr10_successful mixed with ctor initialization list

#include <stdint.h>
#include <cheri/cheric.h>

int nondet_int();

void* foo(void *a) {
  return ((void *)((intptr_t)a & (intptr_t)~1));
}

class t2
{
public:
  int i;

  t2() : i(2) /*init list*/
  {
  }

  int bar();
};


int t2::bar() {
  int a = nondet_int() % 10;
  int *__capability cap_ptr = cheri_ptr(&a, sizeof(a));
  assert(*cap_ptr >= 10 || *cap_ptr < 10);
  return 0;
}

int main() {
  t2 instance2;
  assert(instance2.i == 2);

  instance2.bar();

  return 0;
}
