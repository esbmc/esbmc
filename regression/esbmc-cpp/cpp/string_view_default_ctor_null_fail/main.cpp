// Anti-vacuity twin of string_view_default_ctor_null: data() is null, so it
// cannot also compare equal to a real object's address.
#include <string_view>
#include <cassert>

int main()
{
  char buf[4] = "abc";
  std::string_view sv;
  assert(sv.data() == buf);
  return 0;
}
