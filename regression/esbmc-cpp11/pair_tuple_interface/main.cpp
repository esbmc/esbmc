// [pair.astuple]: pair participates in the tuple protocol, so std::get,
// tuple_size and tuple_element must work on it from <utility> alone --
// <tuple> includes <utility>, not the other way round. Only tuple_element was
// provided, and from the wrong header (issue #5868).
#include <utility>

int read_const(const std::pair<int, int> &p)
{
  return std::get<1>(p);
}

int main()
{
  __ESBMC_assert(
    std::tuple_size<std::pair<int, char>>::value == 2, "a pair is a 2-tuple");

  std::pair<int, int> p(7, 8);
  __ESBMC_assert(std::get<0>(p) == 7 && std::get<1>(p) == 8, "get reads both");

  // get returns a reference, so it writes through to the pair.
  std::get<0>(p) = 9;
  __ESBMC_assert(p.first == 9, "get writes through");
  __ESBMC_assert(read_const(p) == 8, "and has a const overload");

  std::pair<int, char> m(5, 'z');
  std::tuple_element<1, std::pair<int, char>>::type c = std::get<1>(m);
  __ESBMC_assert(c == 'z', "tuple_element names the second type");

  // A structured binding calls get on an rvalue; without that overload the
  // const one wins and the binding fails on the dropped const.
  std::pair<int, int> q(1, 2);
  auto [a, b] = q;
  __ESBMC_assert(a == 1 && b == 2, "structured binding by value");

  auto &[ra, rb] = q;
  ra = 4;
  __ESBMC_assert(q.first == 4, "structured binding by reference writes through");
  return 0;
}
