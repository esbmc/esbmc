#include <string>
#include <cassert>

int main()
{
  // [char.traits.specializations.char]: eq and lt compare as unsigned char, so
  // '\xff' is above 'a', not below it.
  assert(std::char_traits<char>::lt('\xff', 'a'));
  return 0;
}
