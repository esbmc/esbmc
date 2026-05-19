#include <cassert>
#include <variant>

struct A
{
  int x;
};
struct B
{
  double y;
};

int main()
{
  // Primitive alternatives.
  std::variant<int, double> v = 7;
  assert(v.index() == 0);
  assert(std::get<int>(v) == 7);
  assert(std::get<0>(v) == 7);
  assert(std::holds_alternative<int>(v));
  assert(!std::holds_alternative<double>(v));

  v = 3.14;
  assert(v.index() == 1);
  assert(std::holds_alternative<double>(v));

  // Struct alternatives.
  std::variant<A, B> w = A{42};
  assert(std::get<A>(w).x == 42);
  w = B{2.5};
  assert(std::holds_alternative<B>(w));
  assert(std::get<B>(w).y == 2.5);

  // 3-alternative variant + index-based get<>.
  std::variant<int, double, char> z = 'a';
  assert(z.index() == 2);
  assert(std::get<2>(z) == 'a');

  return 0;
}
