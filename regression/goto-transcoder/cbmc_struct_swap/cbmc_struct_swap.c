#include <string.h>
struct P { int a; char b[3]; };
int main()
{
  struct P x = {1, "ab"}, y = {2, "cd"}, t;
  memcpy(&t, &x, sizeof t);
  memcpy(&x, &y, sizeof x);
  memcpy(&y, &t, sizeof y);
  __CPROVER_assert(x.a == 2 && y.a == 1, "struct swap via memcpy");
  __CPROVER_assert(x.b[0] == 'c' && y.b[0] == 'a', "byte members swapped");
  return 0;
}
