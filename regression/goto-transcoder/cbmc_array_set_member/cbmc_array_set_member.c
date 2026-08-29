#include <assert.h>

void __CPROVER_array_set(void *, int);

struct S
{
  int head;
  int a[3];
};

int main(void)
{
  struct S s;
  s.head = 9;
  // CBMC fills the whole object the pointer lands in, so `s.head` is clobbered
  // too. Filling only `s.a` would claim SUCCESSFUL where CBMC reports a
  // violation, so a member array is declined rather than translated.
  __CPROVER_array_set(s.a, 4);
  assert(s.a[1] == 4 && s.head == 9);
  return 0;
}
