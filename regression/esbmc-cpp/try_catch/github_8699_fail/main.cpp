// #8699: the caught pointer reads the thrown literal's characters, so a wrong
// character is reported.
#include <cassert>

int main()
{
  const char *s = nullptr;
  try
  {
    throw "abc";
  }
  catch (const char *p)
  {
    s = p;
  }
  assert(s != nullptr && s[1] == 'c');
  return 0;
}
