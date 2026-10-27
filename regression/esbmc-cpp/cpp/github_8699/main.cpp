// #8699: the operand of throw undergoes array-to-pointer conversion
#include <cassert>
int g[3] = {7, 8, 9};
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
  assert(s != nullptr && s[0] == 'a' && s[2] == 'c');

  int *q = nullptr;
  try
  {
    throw g;
  }
  catch (int *p)
  {
    q = p;
  }
  assert(q == &g[0] && q[2] == 9);
  return 0;
}
