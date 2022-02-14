#include <stdint.h>
#include <cheri/cheric.h>

int nondet_int();

void* foo(void *a) {
  return ((void *)((intptr_t)a & (intptr_t)~1));
}

int main() {
  int a = nondet_int() % 10;
  int *__capability cap_ptr = cheri_ptr(&a, sizeof(a));
  assert(*cap_ptr < 9);
  return 0;
}
