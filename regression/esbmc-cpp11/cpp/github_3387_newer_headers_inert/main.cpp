// <bit> and <span> are C++20 headers and <expected> is C++23.  libstdc++ and
// libc++ expand them to nothing in an older mode rather than erroring, so a
// translation unit that includes them unconditionally still compiles; ESBMC's
// models used to be unguarded and failed to parse (github #3387).
#include <bit>
#include <span>
#include <expected>
#include <array>
#include <cassert>

int main()
{
  std::array<int, 2> a = {{4, 5}};
  assert(a[0] + a[1] == 9);
  return 0;
}
