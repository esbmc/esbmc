#include <cassert>

struct B
{
  int i;
  B() : i(1) {}
};

int main()
{
  B def[2];
  // def[1] IS constructed (i == 1), so this deliberately-wrong assertion
  // must be violated -> VERIFICATION FAILED.
  assert(def[1].i != 1);
  return 0;
}
