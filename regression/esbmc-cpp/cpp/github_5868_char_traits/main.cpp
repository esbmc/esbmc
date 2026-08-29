// esbmc/esbmc#5868: char_traits lived inside <string>, which includes
// <string_view>, so <string_view> could not reach it and the stream OMs only
// forward-declared it. Reaching a member from either header failed to parse.
#include <cassert>
#include <string_view>

int main()
{
  typedef std::char_traits<char> T;
  assert(T::length("abc") == 3);
  assert(T::length("") == 0);
  assert(T::eq('a', 'a'));
  assert(!T::eq('a', 'b'));
  assert(T::lt('a', 'b'));
  assert(T::compare("abc", "abc", 3) == 0);
  assert(T::compare("abc", "abd", 3) < 0);
  assert(T::eof() == -1);
  assert(T::to_int_type('a') == 97);
  assert(T::to_char_type(97) == 'a');
  assert(T::eq_int_type(T::eof(), T::eof()));
  assert(T::not_eof(T::eof()) == 0);
  assert(T::not_eof(5) == 5);
  return 0;
}
