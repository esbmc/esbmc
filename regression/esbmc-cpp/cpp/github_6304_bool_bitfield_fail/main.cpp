// Negative companion to github_6304_bool_bitfield: the bool bitfield holds
// true, so asserting it is false is violated.
#include <cassert>

struct F
{
  bool a : 1;
};

int main()
{
  F f{};
  f.a = 1;
  assert(!f.a); // wrong on purpose: f.a is true
  return 0;
}
