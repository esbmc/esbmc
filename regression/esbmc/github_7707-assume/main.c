#include <stdint.h>

/* The claim is a predicate over the address, not a verdict handed down from the
 * type: a program that constrains its own packed object's base discharges it. */

struct __attribute__((packed)) S
{
  char a;
  char pad[7];
  uint64_t b;
};

int main(void)
{
  struct S sp;
  uint64_t *p = (uint64_t *)&sp.b;
  __ESBMC_assume(((uintptr_t)p & 7u) == 0);
  uint64_t z = *p;
  (void)z;
  return 0;
}
