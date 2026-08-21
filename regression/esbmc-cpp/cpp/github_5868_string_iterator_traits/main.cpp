#include <string>
#include <iterator>
#include <type_traits>
#include <cassert>

int main()
{
  // [iterator.traits] reads these from the iterator itself.
  typedef std::iterator_traits<std::string::iterator> tr;
  static_assert(
    std::is_same<tr::difference_type, std::ptrdiff_t>::value, "difference_type");
  static_assert(std::is_same<tr::value_type, char>::value, "value_type");

  typedef std::iterator_traits<std::string::const_iterator> ctr;
  static_assert(
    std::is_same<ctr::difference_type, std::ptrdiff_t>::value, "const diff");

  std::string s("abc");
  assert(*s.begin() == 'a');
  return 0;
}
