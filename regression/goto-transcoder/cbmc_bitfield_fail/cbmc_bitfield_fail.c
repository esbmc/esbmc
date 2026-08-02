#include <assert.h>
struct S
{
  unsigned a : 3;
};
int main()
{
  struct S s;
  s.a = 9;
  assert(s.a == 9); /* wrong: 9 truncates to 1 in 3 bits */
  return 0;
}