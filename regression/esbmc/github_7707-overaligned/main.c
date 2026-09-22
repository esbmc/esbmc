#include <stdint.h>

/* The check consults alignment(), not "packed implies misaligned": an explicit
 * alignas on a packed struct still constrains its base. */

struct __attribute__((packed, aligned(8))) S
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
