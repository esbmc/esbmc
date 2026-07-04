// A constructor function-try-block whose subobjects do not throw completes
// normally; the implicit rethrow is only reached when an exception is active.
#include <cassert>
int built;
struct Base
{
  Base() { built = 1; }
};
struct Derived : Base
{
  Derived() try : Base() {} catch (int) {}
};
int main()
{
  Derived d;
  assert(built == 1);
  return 0;
}
