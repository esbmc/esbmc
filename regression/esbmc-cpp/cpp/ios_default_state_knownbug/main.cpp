// KNOWNBUG: a freshly constructed stream's ios members are indeterminate.
//
// [basic.ios.cons]: a newly constructed stream has a fill character of
// widen(' '). In this model it does not -- the value is *stable* but
// unconstrained, i.e. read from storage whose initialiser never ran:
//
//   assert(os.fill() == os.fill());   // SUCCEEDS  (stable)
//   assert(os.fill() == ' ');         // FAILS     (uninitialised)
//
// So this is not a nondet-per-call return like the declaration-only defects;
// ios::ios()'s member initialisation simply does not run for a stream built
// through the virtual ios base.
//
// Ruled out while narrowing, so as not to repeat the work: plain virtual-base
// construction works, so does a constructor body that calls its base
// constructor as a discarded temporary (the `ostringstream(){ ostream(); }`
// shape), and so does the combination with polymorphic classes -- each was
// reproduced minimally and verified. ios_base's own state does run, which is
// what makes this specific to the ios level -- and why the exception mask,
// which lives in ios_base, is correctly goodbit here (ios_exceptions_mask).
//
// Setting a value first and reading it back works, which is what the
// ios_exceptions_mask and ios_copyfmt tests assert.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream os;
  assert(os.fill() == ' ');
  return 0;
}
