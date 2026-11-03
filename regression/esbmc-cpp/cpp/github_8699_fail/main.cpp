// #8699: the caught pointer addresses the literal, whose second char is 'b'
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
  assert(s[1] == 'x');
  return 0;
}
