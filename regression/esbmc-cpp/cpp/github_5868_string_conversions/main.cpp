// [string.conversions]: <string> shipped only stoi and stof, and both ignored
// the `pos` out-parameter (github #5868).
#include <string>
#include <cassert>

int main()
{
  assert(std::stoi("42") == 42);
  assert(std::stoi("-7") == -7);
  assert(std::stoi("ff", 0, 16) == 255);
  assert(std::stol("1234") == 1234L);
  assert(std::stoul("40000") == 40000UL);
  assert(std::stoll("-1234") == -1234LL);
  assert(std::stoull("500") == 500ULL);

  assert(std::stof("2.5") == 2.5f);
  assert(std::stod("2.5") == 2.5);
  assert(std::stold("3") == 3.0L);

  // pos receives the index of the first unconverted character.
  size_t p = 0;
  assert(std::stoi("42abc", &p) == 42);
  assert(p == 2);

  p = 0;
  assert(std::stol("  17rest", &p) == 17L);
  assert(p == 4);
  return 0;
}
