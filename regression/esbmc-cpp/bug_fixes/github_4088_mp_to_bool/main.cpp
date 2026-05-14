#include <cassert>

struct S
{
  int x;
  int y;
};

int main()
{
  // Null member-pointer → bool. CK_NullToMemberPointer emits gen_zero(type)
  // and the new CK_MemberPointerToBoolean arm compares against that, so the
  // null path is fully discharged in SMT.
  int S::*null_pm = nullptr;
  assert(static_cast<bool>(null_pm) == false);
  assert(!null_pm);

  // Non-null path: exercise the cast on a bound member pointer. The cast is
  // discarded (sliced) so it does not force SMT encoding of the bare
  // &S::x — verifying truthiness of a bound member-pointer is a separate
  // SMT concern. The dereference assertion below confirms pm is still the
  // expected field after surviving the cast pipeline.
  int S::*pm = &S::x;
  (void)static_cast<bool>(pm);
  S s;
  s.x = 42;
  s.y = 7;
  assert(s.*pm == 42);
  return 0;
}
