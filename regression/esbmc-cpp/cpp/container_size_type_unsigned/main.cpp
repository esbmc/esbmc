#include <vector>
#include <set>
#include <type_traits>
#include <cassert>

int main()
{
  // [container.requirements]: size_type is an unsigned integer type.
  static_assert(
    std::is_unsigned<std::vector<int>::size_type>::value, "vector size_type");
  static_assert(
    std::is_unsigned<std::set<int>::size_type>::value, "set size_type");

  // The observable consequence: size() - 1 on an empty container wraps.
  std::vector<int> v;
  assert(v.size() - 1 > 1000);

  std::set<int> s;
  assert(s.size() - 1 > 1000);

  // Ordinary use is unchanged.
  v.push_back(1);
  v.push_back(2);
  assert(v.size() == 2);
  assert(v.size() - 1 == 1);
  s.insert(3);
  assert(s.size() == 1);
  return 0;
}
