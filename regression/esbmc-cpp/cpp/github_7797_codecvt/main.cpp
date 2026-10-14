// #7797: <codecvt> must be includable and carry the facet constants.
#include <codecvt>
#include <cassert>

int main()
{
  assert(std::consume_header == 4);
  assert(std::generate_header == 2);
  assert(std::little_endian == 1);

  // max_length() values measured from libc++: UTF-8 needs 3 units for a
  // char16_t element and 4 otherwise, UTF-16 needs 2 and 4, and consuming a
  // header adds one sequence.
  assert(std::codecvt_utf8<char32_t>().max_length() == 4);
  assert(std::codecvt_utf8<char16_t>().max_length() == 3);
  assert(std::codecvt_utf8<wchar_t>().max_length() == 4);
  assert(
    (std::codecvt_utf8<char32_t, 0x10ffff, std::consume_header>()
       .max_length()) == 7);

  assert(std::codecvt_utf16<char32_t>().max_length() == 4);
  assert(std::codecvt_utf16<char16_t>().max_length() == 2);
  assert(
    (std::codecvt_utf16<char16_t, 0x10ffff, std::consume_header>()
       .max_length()) == 4);

  assert(std::codecvt_utf8_utf16<char32_t>().max_length() == 4);
  assert(std::codecvt_utf8_utf16<char16_t>().max_length() == 4);
  assert(
    (std::codecvt_utf8_utf16<char32_t, 0x10ffff, std::consume_header>()
       .max_length()) == 7);

  assert(std::codecvt_utf8<char32_t>().encoding() == 0);
  assert(!std::codecvt_utf8<char32_t>().always_noconv());
  return 0;
}
