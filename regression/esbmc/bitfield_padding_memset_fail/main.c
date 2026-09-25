#include <string.h>
#include <assert.h>

/* Anti-vacuity for the bitfield-padding pins: the object really is
   0x000FFFFF, so an off-by-one claim about it must still be refuted. */
typedef struct
{
  int x : 12, y : 8;
} S;

int main()
{
  S s;
  memset(&s, 0, sizeof(s));
  s.x = -1;
  s.y = -1;
  unsigned v = (unsigned)*(int *)&s;
  assert(v == 0x000FFFFEu);
  return 0;
}
