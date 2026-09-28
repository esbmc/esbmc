// string_view::at, starts_with, ends_with, find, find_first_of and
// find_first_not_of were DECLARED but never DEFINED. A declaration-only member
// converts to a call returning a nondeterministic value, so on "hello" both
// `assert(v.starts_with("he"))` and `assert(!v.starts_with("he"))` FAILED --
// a silent wrong answer rather than a diagnostic.
//
// Verified against clang++ -std=c++20 -fsanitize=address,undefined: exits 0.
#include <string_view>
#include <cassert>
int main()
{
  std::string_view v("hello world");
  std::string_view he("he"), ld("ld"), xx("xx"), wo("wo"), empty("");

  assert(v.starts_with(he));
  assert(!v.starts_with(ld));
  assert(v.ends_with(ld));
  assert(!v.ends_with(he));
  assert(v.starts_with(empty));
  assert(v.ends_with(empty));

  assert(v.find(wo) == 6);
  assert(v.find(he) == 0);
  assert(v.find(xx) == std::string_view::npos);
  assert(v.find(wo, 7) == std::string_view::npos);

  std::string_view vowels("aeiou");
  assert(v.find_first_of(vowels) == 1);                      // 'e' in "hello"
  assert(v.find_first_not_of(std::string_view("hel")) == 4); // 'o'

  // a needle longer than the haystack
  std::string_view longer("hello world!!");
  assert(!v.starts_with(longer));
  assert(!v.ends_with(longer));
  assert(v.find(longer) == std::string_view::npos);

  assert(v.at(0) == 'h');
  assert(v.at(10) == 'd');
  return 0;
}
