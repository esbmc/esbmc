// [support.types.layout]/1 gives offsetof the semantics of the C macro, which
// C23 7.21p3 requires to be an integer constant expression. ESBMC's
// __builtin_offsetof expansion is a pointer computation, so under it every
// constexpr, static_assert and non-type template argument spelt with offsetof
// was rejected -- including the ones immer's node layout is built from, which
// is what stopped ESBMC parsing its own irep2 headers.
#include <cstddef>
#include <cassert>

struct layout
{
  char a;
  int b;
  double c;
};

static_assert(offsetof(layout, b) > 0, "offsetof is a constant expression");

template <std::size_t N>
struct at_offset
{
  static constexpr std::size_t value = N;
};

constexpr std::size_t off_b = offsetof(layout, b);
constexpr std::size_t off_c = offsetof(layout, c);

int main()
{
  layout l;
  char *base = reinterpret_cast<char *>(&l);

  assert(at_offset<off_b>::value == off_b);
  assert(off_b < off_c);
  assert(reinterpret_cast<char *>(&l.b) - base == (long)off_b);
  assert(reinterpret_cast<char *>(&l.c) - base == (long)off_c);
  return 0;
}
