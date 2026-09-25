// <ostream> declared its own width/fill/precision members. [ostream] has no
// such members -- they belong to ios_base/ios -- and declaring them here hid
// the inherited overload sets:
//
//   * cout.precision() did not resolve at all ("too few arguments"), because
//     the void-returning one-argument ostream::precision hid both base
//     overloads;
//   * cout.precision(3) and cout.fill(c) bound to stubs whose bodies were
//     commented out, so the setting was silently discarded;
//   * ostream::width was declared and never defined, so it returned a
//     nondeterministic value.
//
// Removing all four restores the inherited members, which are implemented.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream os;

  // precision: the setter returns the previous value, the getter observes it
  os.precision(3);
  assert(os.precision() == 3);
  std::streamsize prev = os.precision(5);
  assert(prev == 3);
  assert(os.precision() == 5);

  // width
  os.width(7);
  assert(os.width() == 7);

  // fill
  os.fill('0');
  assert(os.fill() == '0');
  char oldfill = os.fill('x');
  assert(oldfill == '0');
  assert(os.fill() == 'x');

  return 0;
}
