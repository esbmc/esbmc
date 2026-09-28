// esbmc/esbmc#7048 negative control: char_traits<char>::eq and lt must behave as the built-in
// == and < for unsigned char ([char.traits.specializations.char]), so every
// char above 0x7f orders above the ASCII range. compare() is built on lt, so
// std::string's ordering inherits whichever answer these give.
#include <cassert>
#include <string>

int main()
{
  typedef std::char_traits<char> T;
  const char hi = (char)0x80;
  const char top = (char)0xff;

  assert(!T::lt('a', hi));
  assert(!T::lt(hi, 'a'));
  assert(T::lt(hi, top));
  assert(!T::lt(top, hi));
  assert(!T::lt('a', 'a'));

  assert(T::eq(hi, hi));
  assert(!T::eq(hi, 'a'));

  const char lo_s[2] = {'a', 0};
  const char hi_s[2] = {hi, 0};
  assert(T::compare(lo_s, hi_s, 1) < 0);
  assert(T::compare(hi_s, lo_s, 1) > 0);
  assert(T::compare(hi_s, hi_s, 1) == 0);
  return 0;
}
