#include <string>
#include <cassert>

int main()
{
  std::string s = "hello";

  // C-string overload
  assert(s.starts_with("he") && !s.starts_with("lo") && !s.starts_with("hellox"));
  assert(s.ends_with("lo") && !s.ends_with("he") && !s.ends_with("xhello"));

  // char overload
  assert(s.starts_with('h') && !s.starts_with('e'));
  assert(s.ends_with('o') && !s.ends_with('h'));

  // basic_string overload
  std::string p = "hel";
  assert(s.starts_with(p) && !s.starts_with(std::string("xyz")));

  // full match and empty prefix/suffix
  assert(s.starts_with("hello") && s.ends_with("hello"));
  assert(s.starts_with("") && s.ends_with(""));

  // empty string
  std::string e = "";
  assert(e.starts_with("") && e.ends_with("") && !e.starts_with("x"));

  return 0;
}
