// [streambuf]: streambuf is basic_streambuf<char>, not a class of its own.
// Code that names the template -- boost's lexical_cast derives its unlocked
// buffer from std::basic_streambuf<CharT, Traits> -- could not compile while
// only the char instantiation existed. See #5868.
#include <streambuf>
#include <cassert>

template <class CharT, class Traits>
struct probe : std::basic_streambuf<CharT, Traits>
{
  typedef typename std::basic_streambuf<CharT, Traits>::char_type char_type;
  typedef typename std::basic_streambuf<CharT, Traits>::int_type int_type;
  typedef typename std::basic_streambuf<CharT, Traits>::traits_type traits_type;
};

typedef probe<char, std::char_traits<char> > narrow;
typedef probe<wchar_t, std::char_traits<wchar_t> > wide;

int main()
{
  narrow n;
  wide w;
  (void)n;
  (void)w;

  narrow::char_type c = 'a';
  assert(c == 'a');
  assert(sizeof(narrow::char_type) == sizeof(char));
  assert(sizeof(wide::char_type) == sizeof(wchar_t));

  // streambuf names the char instantiation, so a pointer to one converts.
  std::streambuf *sb = static_cast<std::basic_streambuf<char> *>(0);
  assert(sb == 0);
  return 0;
}
