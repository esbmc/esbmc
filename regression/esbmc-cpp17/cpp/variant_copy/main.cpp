// Copying a variant used to lose the active alternative. The converting
// constructor template was unconstrained, so for a non-const lvalue it beat the
// implicit copy constructor; __assign then matched no alternative and left the
// target at index 0 with an empty slot -- a silent wrong answer.
// [variant.ctor]/13 requires that constructor not to participate when the
// argument is the variant itself.
#include <variant>
#include <cassert>

int main()
{
  std::variant<int, double> v = 2.5;
  assert(v.index() == 1);

  std::variant<int, double> w = v; // copy construction
  assert(w.index() == 1);
  assert(std::get<double>(w) == 2.5);
  assert(std::holds_alternative<double>(w));

  std::variant<int, double> x = 7;
  assert(x.index() == 0);
  x = w; // copy assignment
  assert(x.index() == 1);
  assert(std::get<double>(x) == 2.5);

  std::variant<int, double> y = std::move(w); // move construction
  assert(y.index() == 1);
  assert(std::get<double>(y) == 2.5);

  // A converting assignment still selects the matching alternative.
  y = 3;
  assert(y.index() == 0);
  assert(std::get<int>(y) == 3);

  return 0;
}
