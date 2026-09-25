#include <cassert>
#include <cwchar>

int main()
{
  wchar_t buf[4] = {L'a', L'b', L'c', L'\0'};

  // The third element is 'c', not 'a'.
  assert(buf[2] == L'a');

  return 0;
}
