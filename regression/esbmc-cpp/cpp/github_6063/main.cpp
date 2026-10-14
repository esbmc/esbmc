// STL model gaps surfaced by <boost/program_options.hpp> (github #6063):
// string::size_type, const map::find, std::wstring, <iosfwd>, and the
// std-namespace aliases of <cstring>/<cstdio>, plus is_const /
// is_member_pointer from <type_traits>.
#include <iosfwd>
#include <string>
#include <map>
#include <cstring>
#include <cstdio>
#include <type_traits>
#include <cassert>

typedef std::wstring *wstring_is_declared;

int lookup(const std::map<int, int> &m)
{
  return m.find(1)->second;
}

struct S
{
  int m;
};

int main()
{
  std::string s("abc");
  std::string::size_type i = s.find('b');
  assert(i == 1);

  std::map<int, int> m;
  m[1] = 42;
  assert(lookup(m) == 42);

  assert(std::strlen("hello") == 5);
  assert(std::strcmp("a", "a") == 0);

  static_assert(std::is_const<const int>::value, "is_const");
  static_assert(std::is_member_pointer<int S::*>::value, "is_member_pointer");
  return 0;
}
