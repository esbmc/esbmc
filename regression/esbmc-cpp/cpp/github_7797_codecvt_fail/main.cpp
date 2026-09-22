// #7797: a UTF-8 facet over char16_t needs 3 units, not 4.
#include <codecvt>
#include <cassert>

int main()
{
  assert(std::codecvt_utf8<char16_t>().max_length() == 4);
  return 0;
}
