// Negative counterpart of array_member_empty_struct: the array-literal
// assignment for the empty-class array member must succeed (no
// "assignment to constant_array not handled" abort) so that a genuine
// property violation after it is still reported as VERIFICATION FAILED.
#include <cassert>

struct E
{
};

struct C
{
  E a[3];
};

int main()
{
  C c;
  (void)c;
  assert(0);
  return 0;
}
