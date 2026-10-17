// #8699: the operand of throw undergoes the array-to-pointer conversion, so a
// string literal is caught as a const char * pointing at its characters.
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
  assert(s != nullptr && s[0] == 'a' && s[3] == '\0');
  return 0;
}
