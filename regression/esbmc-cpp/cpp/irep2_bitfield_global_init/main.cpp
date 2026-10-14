#include <cassert>
#include <string.h>

// A bitfield member's legacy type carries `#bitfield` and its underlying type,
// neither of which crosses the migrate seam: add_padding applied to a
// back-migrated type sees plain narrow integers, inserts no bit-field pad, and
// the zero-initialiser of this global comes out a member short -- so the object
// is 6 bytes where memset writes 8.
struct has_bitfield
{
  unsigned int a;
  unsigned int b : 2;
  unsigned int c : 2;
  unsigned int d : 2;
} beans;

int main()
{
  assert(beans.a == 0 && beans.b == 0 && beans.c == 0 && beans.d == 0);
  memset(&beans, 0, sizeof(beans));
  assert(beans.d == 0);
}
