// visit dispatches on the alternative actually held, so asserting the answer
// for a different one stays refutable.
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
};

int main()
{
  std::variant<int, double> v = 2.5;
  assert(std::visit(ToInt(), v) == 2);
  return 0;
}
