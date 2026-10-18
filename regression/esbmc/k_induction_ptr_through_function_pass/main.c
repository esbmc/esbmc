// Companion to k_induction_ptr_through_function_fail.
//
// Simple sanity: a k-induction run with no loops at all — just a
// straight-line program with an assertion. Confirms the fix doesn't
// crash or regress basic cases.
//
// Expected: VERIFICATION SUCCESSFUL.

#include <assert.h>

int main(void)
{
  int x = 42;
  assert(x == 42);
  return 0;
}
