#include <assert.h>
int main()
{
  int a, b;
  __CPROVER_assume(a == 2 && b == 3); // 2 + 3 = 5, no overflow
  assert(__builtin_add_overflow_p(a, b, (int)0)); // claims overflow -> FAILED
  return 0;
}
