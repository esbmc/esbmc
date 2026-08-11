// The same discarded-result shape for alloca, which has the enclosing
// function's lifetime and so must not be reported as a leak. Pins the other
// half of github_822_malloc_no_lhs_fail. #822
#include <alloca.h>

int main()
{
  alloca(10);
  return 0;
}
