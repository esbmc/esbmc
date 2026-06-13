#include <cassert>

extern "C" int multi_tu_first();
extern "C" int multi_tu_second();

int main()
{
  // The second TU's body is now correctly merged, so multi_tu_second()
  // returns exactly 2. Asserting it equals 3 must therefore fail with a
  // concrete counterexample (rather than passing vacuously on a nondet body).
  assert(multi_tu_first() == 1);
  assert(multi_tu_second() == 3);
  return 0;
}
