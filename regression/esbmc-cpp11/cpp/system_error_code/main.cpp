// <system_error> was not bundled, so any program including it failed to parse
// (github #5868 lists it among the missing headers). The category identity is
// the load-bearing part: two codes with the same value but different categories
// must not compare equal.
#include <system_error>
#include <cassert>
#include <cstring>

int main()
{
  std::error_code e;
  assert(!e);
  assert(e.value() == 0);

  std::error_code f = std::make_error_code(std::errc::invalid_argument);
  assert(f);
  assert(f.value() == 22);
  assert(f.category() == std::generic_category());
  assert(std::strcmp(f.category().name(), "generic") == 0);

  std::error_code g(22, std::system_category());
  assert(g.value() == 22);
  assert(g.category() != f.category());
  assert(g != f); // same value, different category

  std::error_code h(22, std::generic_category());
  assert(h == f);

  std::error_code i = std::errc::io_error; // implicit errc conversion
  assert(i.value() == 5);

  f.clear();
  assert(!f);
  assert(f.value() == 0);

  return 0;
}
