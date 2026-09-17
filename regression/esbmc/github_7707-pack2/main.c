#include <stdint.h>

/* `#pragma pack(n)` reaches the type as `max_field_alignment`, which does not
 * survive migration to irep2 -- the base alignment has to be read off the
 * legacy type or this access reads as naturally aligned. */

#pragma pack(2)
struct S
{
  char a;
  char pad[3];
  uint32_t b;
};
#pragma pack()

int main(void)
{
  struct S sp;
  uint32_t *p = (uint32_t *)&sp.b;
  uint32_t z = *p;
  (void)z;
  return 0;
}
