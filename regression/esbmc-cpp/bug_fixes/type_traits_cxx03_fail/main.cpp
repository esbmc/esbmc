// <type_traits> is pulled in unconditionally by <memory>, <string>, <cmath>
// and friends, so it must parse under -std=c++03. Exercise the traits that
// C++03 can actually name (no alias or variable templates).
#include <string>
#include <stdexcept>
#include <memory>
#include <cmath>
#include <type_traits>
#include <cassert>

template <typename T>
typename std::enable_if<std::is_integral<T>::value, int>::type only_integral(T)
{
  return 1;
}

int main()
{
  assert(std::is_integral<int>::value);
  assert(!std::is_integral<double>::value);
  assert(std::is_floating_point<double>::value);
  assert(std::is_arithmetic<char>::value);
  assert(std::is_pointer<int *>::value);
  assert(!std::is_reference<int>::value);
  assert(std::is_lvalue_reference<int &>::value);
  assert(!std::is_rvalue_reference<int>::value);
  assert(std::extent<int[5]>::value == 6); // negative variant: 5, not 6

  assert((std::is_same<int, int>::value));
  assert(!(std::is_same<int, long>::value));
  assert((std::is_same<std::remove_cv<const int>::type, int>::value));
  assert((std::is_same<std::make_unsigned<int>::type, unsigned int>::value));
  assert((std::is_same<std::make_signed<unsigned int>::type, int>::value));
  assert((std::is_same<std::decay<const int &>::type, int>::value));
  assert((std::is_same<std::conditional<true, int, long>::type, int>::value));

  assert(only_integral(3) == 1);
  return 0;
}
