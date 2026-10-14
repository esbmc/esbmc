#include <cassert>
#include <string.h>

// An over-aligned empty struct occupies its alignment, so its padded layout has
// a trailing pad member and a literal of it must carry an operand for it. The
// alignment reaches IREP2 only if the seam carries it: without it add_padding
// sees no alignment, the literal stays shorter than its own type, and the SMT
// tuple layer reads past the end of its element vector.
struct alignas(16) E
{
};

int main()
{
  E e = {};
  char zeroes[sizeof(E)] = {};
  assert(sizeof(E) == 16);
  assert(!memcmp(&e, &zeroes, sizeof(E)));
}
