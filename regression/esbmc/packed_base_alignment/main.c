#include <stdint.h>

/* A packed object's base is left unconstrained by the address-space model, so
   a pointer laundered out of a packed member is not known to be aligned. The
   offset-only alignment check used to pass this (#7707); UBSan traps on it. */
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
  uint64_t z = *p;
  (void)z;
  return 0;
}
