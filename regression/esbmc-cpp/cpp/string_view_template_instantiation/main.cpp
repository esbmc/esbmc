// [string.view.template]: string_view is basic_string_view<char>, not a class
// of its own. Code that names the template -- boost's lexical_cast specialises
// stream_char_common on std::basic_string_view<Char, Traits> -- could not
// compile while only the char instantiation existed. See #5868.
#include <string_view>
#include <cassert>

template <class SV>
size_t view_size(SV sv)
{
  return sv.size();
}

int main()
{
  std::basic_string_view<char> bsv("hello");
  assert(bsv.size() == 5);
  assert(bsv[1] == 'e');
  assert(view_size(bsv) == 5);

  std::string_view sv("hello world");
  assert(sv.size() == 11);
  assert(sv.find(std::string_view("wor")) == 6);
  assert(sv.starts_with(std::string_view("hell")));
  assert(sv.ends_with(std::string_view("rld")));

  std::wstring_view wsv(L"abcd");
  assert(wsv.size() == 4);
  assert(wsv[3] == L'd');
  assert(view_size(wsv) == 4);

  std::u16string_view u16(u"xyz");
  assert(u16.size() == 3);
  assert(u16[0] == u'x');
  return 0;
}
