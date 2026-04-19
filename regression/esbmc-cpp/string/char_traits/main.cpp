#include <cassert>
#include <cstddef>
#include <cstdio>   // for EOF
#include <string>

int main() {
  using traits = std::char_traits<char>;

  // Test type conversions
  char ch = 'x';
  traits::int_type i = traits::to_int_type(ch);
  assert(i == static_cast<int>(ch));

  char ch2 = traits::to_char_type(i);
  assert(ch2 == ch);

  // eq_int_type
  assert(traits::eq_int_type(i, traits::to_int_type(ch2)));

  // eof() and not_eof()
  traits::int_type eof_val = traits::eof();
  assert(eof_val == EOF);
  assert(traits::not_eof(eof_val) == 0);
  assert(traits::not_eof(i) == i);

  // eq, lt
  assert(traits::eq('a', 'a'));
  assert(!traits::eq('a', 'b'));
  assert(traits::lt('a', 'b'));
  assert(!traits::lt('b', 'a'));

  // compare
  const char *s1 = "abc";
  const char *s2 = "abc";
  const char *s3 = "abd";
  assert(traits::compare(s1, s2, 3) == 0);
  assert(traits::compare(s1, s3, 3) < 0);
  assert(traits::compare(s3, s1, 3) > 0);

  // length
  assert(traits::length("hello") == 5);
  assert(traits::length("") == 0);

  // find
  const char *found = traits::find("hello", 5, 'e');
  assert(found != nullptr);
  assert(*found == 'e');
  assert(traits::find("hello", 5, 'x') == nullptr);

  return 0;
}
