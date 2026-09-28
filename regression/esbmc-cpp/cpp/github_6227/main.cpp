#include <cassert>
#include <string>

int main()
{
  std::string a = "AAAA", b = "AA", c = "AAAA", d = "AB";

  // Prefix relations: a longer string sharing the shorter one's prefix is
  // greater. These are exactly the cases the length-ignoring loop got wrong.
  assert(a > b);
  assert(b < a);
  assert(!(b > a));
  assert(!(a < b));

  // Equal strings are neither greater nor less, but compare >= and <=.
  assert(!(a > c));
  assert(!(a < c));
  assert(a >= c);
  assert(a <= c);

  // A larger character outweighs a longer length.
  assert(d > a); // "AB" > "AAAA"
  assert(a < d);

  // const operands are not viable for the non-const member operators, so these
  // exercise the free operator>/operator< overloads.
  const std::string ca = "AAAA", cb = "AA";
  assert(ca > cb);
  assert(cb < ca);

  return 0;
}
