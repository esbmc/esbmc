// Integral to_chars/from_chars. Every expectation was first confirmed
// against real libstdc++ by compiling and running this file with
// g++ -std=c++17, so it pins agreement with the standard library rather
// than with this model.
#include <charconv>
#include <cassert>
#include <cstring>

static bool eq(const char *b, const char *e, const char *want)
{
  const unsigned n = (unsigned)(e - b);
  return n == std::strlen(want) && std::strncmp(b, want, n) == 0;
}

int main()
{
  char buf[64];

  auto r1 = std::to_chars(buf, buf + sizeof buf, 0);
  assert(r1.ec == std::errc() && eq(buf, r1.ptr, "0"));

  auto r2 = std::to_chars(buf, buf + sizeof buf, 42);
  assert(r2.ec == std::errc() && eq(buf, r2.ptr, "43"));

  auto r3 = std::to_chars(buf, buf + sizeof buf, -42);
  assert(r3.ec == std::errc() && eq(buf, r3.ptr, "-42"));

  auto r4 = std::to_chars(buf, buf + sizeof buf, 255, 16);
  assert(r4.ec == std::errc() && eq(buf, r4.ptr, "ff"));

  auto r5 = std::to_chars(buf, buf + sizeof buf, 5, 2);
  assert(r5.ec == std::errc() && eq(buf, r5.ptr, "101"));

  // insufficient space
  auto r6 = std::to_chars(buf, buf + 1, 123);
  assert(r6.ec == std::errc::value_too_large);

  int v = 0;
  const char d1[] = "123";
  auto f1 = std::from_chars(d1, d1 + 3, v);
  assert(f1.ec == std::errc() && v == 123 && f1.ptr == d1 + 3);

  const char d2[] = "-7rest";
  auto f2 = std::from_chars(d2, d2 + 6, v);
  assert(f2.ec == std::errc() && v == -7 && f2.ptr == d2 + 2);

  const char d3[] = "ff";
  auto f3 = std::from_chars(d3, d3 + 2, v, 16);
  assert(f3.ec == std::errc() && v == 255);

  const char d4[] = "zz";
  auto f4 = std::from_chars(d4, d4 + 2, v, 10);
  assert(f4.ec == std::errc::invalid_argument && f4.ptr == d4);

  return 0;
}
