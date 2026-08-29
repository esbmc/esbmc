// [func.bind.place]: _1..._N have distinct unspecified types, and
// is_placeholder<decltype(_n)>::value == n. Distinct *types* is what matters:
// boost/bind/std_placeholders.hpp partially specialises its own is_placeholder
// on each of the first nine, which collapses to a redefinition if they share a
// type. See #5868.
#include <functional>
#include <type_traits>
#include <cassert>

template <class T>
struct one_per_type
{
  enum
  {
    value = 0
  };
};
template <>
struct one_per_type<std::decay<decltype(std::placeholders::_1)>::type>
{
  enum
  {
    value = 1
  };
};
template <>
struct one_per_type<std::decay<decltype(std::placeholders::_2)>::type>
{
  enum
  {
    value = 2
  };
};
template <>
struct one_per_type<std::decay<decltype(std::placeholders::_9)>::type>
{
  enum
  {
    value = 9
  };
};

int main()
{
  assert(one_per_type<std::decay<decltype(std::placeholders::_1)>::type>::value == 1);
  assert(one_per_type<std::decay<decltype(std::placeholders::_2)>::type>::value == 2);
  assert(one_per_type<std::decay<decltype(std::placeholders::_9)>::type>::value == 9);
  assert(one_per_type<int>::value == 0);

  assert(std::is_placeholder<std::decay<decltype(std::placeholders::_3)>::type>::value == 3);
  assert(std::is_placeholder<std::decay<decltype(std::placeholders::_10)>::type>::value == 10);
  assert(std::is_placeholder<int>::value == 0);
  return 0;
}
