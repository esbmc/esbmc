// Companion to dangling_temporary_ref: a temporary bound DIRECTLY to a
// const reference has its lifetime extended to that of the reference
// ([class.temporary]/6), so these reads are valid and must keep verifying.
// A fix for the dangling case that simply kills every temporary at the end of
// the full expression would break this.
#include <cassert>

struct S
{
  int v;
  ~S()
  {
  }
};

int main()
{
  const int &r = 4;
  assert(r == 4);

  const S &s = S{7};
  assert(s.v == 7);

  return 0;
}
