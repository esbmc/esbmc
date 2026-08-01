// KNOWNBUG: a reference to a temporary that has already died is not detected --
// ESBMC reports SUCCESSFUL where the read is a genuine use-after-lifetime.
//
// [class.temporary]/4: the temporaries created for the arguments are destroyed
// at the end of the full-expression, and [class.temporary]/6 does NOT extend
// their lifetime here because the reference binds to the function's *return*
// value, not directly to a temporary. clang++ -fsanitize=address aborts on this
// program; ESBMC verifies it.
//
// The GOTO shows why: the argument temporaries are declared in main's scope and
// never marked dead, so they outlive the full expression.
//
//     DECL signed int tmp$1;  ASSIGN tmp$1=4;
//     DECL signed int tmp$2;  ASSIGN tmp$2=2;
//     FUNCTION_CALL: return_value$_pick$3=pick(&tmp$1, &tmp$2)
//     ASSIGN r=return_value$_pick$3;
//     -- no DEAD tmp$1 / DEAD tmp$2 here --
//
// This is the shape std::min/std::max/std::minmax produce, so it is a common
// real-world dangling-reference bug.
//
// CAUTION for whoever fixes it: emitting DEAD for every argument temporary at
// the end of the full expression would be wrong. A temporary bound *directly*
// to a reference does have its lifetime extended ([class.temporary]/6) --
// see temporary_lifetime_extension, which must keep verifying.
//
// Detection does work for the neighbouring shapes; both of these are correctly
// reported as failures today, which is what localises the gap:
//   * a pointer to a block-scope local read after the block ends;
//   * a function returning a reference to its own local.
#include <cassert>

static const int &pick(const int &a, const int &b)
{
  return a < b ? a : b;
}

int main()
{
  const int &r = pick(4, 2);
  assert(r == 2);
  return 0;
}
