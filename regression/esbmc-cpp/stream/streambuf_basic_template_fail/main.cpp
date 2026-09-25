// Anti-vacuity twin of streambuf_basic_template: a non-char instantiation has
// to carry its own character type, not char's.
#include <streambuf>
#include <cassert>

template <class CharT, class Traits>
struct probe : std::basic_streambuf<CharT, Traits>
{
  typedef typename std::basic_streambuf<CharT, Traits>::char_type char_type;
};

int main()
{
  assert(
    sizeof(probe<wchar_t, std::char_traits<wchar_t> >::char_type) ==
    sizeof(char));
  return 0;
}
