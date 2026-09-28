// Anti-vacuity twin of string_view_template_instantiation: a non-char
// instantiation has to measure its own character type, not bytes.
#include <string_view>
#include <cassert>

int main()
{
  std::wstring_view wsv(L"abcd");
  assert(wsv.size() == sizeof(wchar_t) * 4);
  return 0;
}
