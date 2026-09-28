// github #5868, negative direction: the string -> string_view conversion must
// carry the real length, so a wrong expected size has to be caught. Without
// this the checks in github_5868_string_view_hash could pass vacuously.
#include <string>
#include <string_view>
#include <cassert>

int main()
{
  std::string s = "hello";
  std::string_view v = s;
  assert(v.size() == 4);
  return 0;
}
