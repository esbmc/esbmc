#include <cassert>
#include <type_traits>

struct Functor
{
  double operator()(int) const;
};

int main()
{
  // invoke_result_t<Functor, int> is double, so this comparison is false.
  bool same = std::is_same<std::invoke_result_t<Functor, int>, int>::value;
  assert(same);

  return 0;
}
