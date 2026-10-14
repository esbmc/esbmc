// std::visit calls the visitor on the alternative the variant currently holds.
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cassert>
#include <variant>

struct ToInt
{
  int operator()(int x) const
  {
    return x;
  }
  int operator()(double d) const
  {
    return (int)d + 100;
  }
  int operator()(char c) const
  {
    return (int)c + 1000;
  }
};

static int calls = 0;

struct Counting
{
  void operator()(int)
  {
    calls++;
  }
  void operator()(double)
  {
    calls++;
  }
};

int main()
{
  std::variant<int, double, char> v;

  v = 5;
  assert(std::visit(ToInt(), v) == 5);
  v = 2.5;
  assert(std::visit(ToInt(), v) == 102);
  v = 'a';
  assert(std::visit(ToInt(), v) == 1097);

  const std::variant<int, double, char> &cv = v;
  assert(std::visit(ToInt(), cv) == 1097);

  // Two alternatives: the storage's surplus slots must not be instantiated.
  std::variant<int, double> w = 7;
  assert(std::visit(ToInt(), w) == 7);

  // A void visitor runs exactly once per visit, on every alternative.
  std::variant<int, double> u = 1;
  std::visit(Counting(), u);
  u = 2.0;
  std::visit(Counting(), u);
  assert(calls == 2);
  return 0;
}
