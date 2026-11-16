#include <cassert>

/* #10008: a C++20 parenthesised aggregate init of a union names its member. */
union U
{
  int a;
  float b;
};

int main()
{
  U u(17);
  assert(u.a == 18);
  return 0;
}
