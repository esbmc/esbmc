#include <assert.h>
struct S
{
  _Bool a : 1;
  _Bool b : 1;
};
int main()
{
  struct S s; // nondet
  _Bool r = s.a || s.b;
  assert(!r);
  return 0;
}
