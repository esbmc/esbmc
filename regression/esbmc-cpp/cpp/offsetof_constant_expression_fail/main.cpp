// Anti-vacuity twin of offsetof_constant_expression: the constant folded at
// parse time has to be the member's real offset, not just some constant.
#include <cstddef>
#include <cassert>

struct layout
{
  char a;
  int b;
  double c;
};

constexpr std::size_t off_b = offsetof(layout, b);

int main()
{
  layout l;
  char *base = reinterpret_cast<char *>(&l);
  assert(reinterpret_cast<char *>(&l.c) - base == (long)off_b);
  return 0;
}
