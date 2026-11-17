#include <stdint.h>
#include <string.h>

/* A struct whose bytes are copied from a nondet integer holds a pointer that
 * can name any object, b included, with or without slicing (#10798). The
 * sliced run misses it: the copy reaches the struct as a bitcast, which is
 * also how every write through a pointer into a struct is encoded, so
 * treating it as an int-to-pointer cast exhausted memory on
 * esbmc-cpp/cpp/ch9_7. */
uintptr_t nondet_uintptr(void);
struct S
{
  int *p;
};
int main(void)
{
  int a = 1, b = 2;
  int *q = &a;
  (void)q;
  uintptr_t i = nondet_uintptr();
  struct S s;
  memcpy(&s, &i, sizeof s);
  __ESBMC_assert(
    __ESBMC_POINTER_OBJECT(s.p) <= 1 || __ESBMC_same_object(s.p, &b),
    "s.p names null, invalid or b");
  return 0;
}
