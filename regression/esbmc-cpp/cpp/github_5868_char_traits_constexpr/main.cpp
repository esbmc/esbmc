#include <string>
#include <cassert>

// nlohmann/json wraps char_traits in exactly this shape.
struct wrapped
{
  typedef int int_type;
  static constexpr int_type eof() noexcept
  {
    return static_cast<int_type>(std::char_traits<char>::eof());
  }
};

int main()
{
  // [char.traits.specializations.char]: these are constexpr since C++11.
  static_assert(wrapped::eof() == std::char_traits<char>::eof(), "eof");
  static_assert(std::char_traits<char>::eq('a', 'a'), "eq");
  static_assert(!std::char_traits<char>::lt('b', 'a'), "lt");
  static_assert(std::char_traits<char>::to_int_type('a') == 97, "to_int_type");
  static_assert(std::char_traits<char>::to_char_type(97) == 'a', "to_char_type");
  static_assert(std::char_traits<char>::eq_int_type(1, 1), "eq_int_type");
  static_assert(std::char_traits<char>::not_eof(97) == 97, "not_eof");
  static_assert(std::char_traits<char>::not_eof(EOF) == 0, "not_eof at eof");

  // The unsigned-char ordering of #7048 must survive the constexpr change.
  assert(std::char_traits<char>::lt('a', '\xff'));
  return 0;
}
