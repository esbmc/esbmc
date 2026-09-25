// The same unbound-operand defect reached defaulted operator<=> declared as a
// friend, which aborted in member2t rather than returning a wrong answer.
#include <compare>
#include <cassert>

struct S
{
  int v;
  friend auto operator<=>(S, S) = default;
  friend bool operator==(S, S) = default;
};

int main()
{
  S a{1}, b{2};
  assert((a <=> b) < 0);
  assert((b <=> a) > 0);
  assert((a <=> a) == 0);
  assert(a < b);
  assert(!(b < a));
  return 0;
}
