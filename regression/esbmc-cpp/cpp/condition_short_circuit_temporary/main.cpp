// A temporary in an operand of &&, || or ?: in a condition lives until the end
// of the condition, a full-expression ([class.temporary]/4), and only if its
// operand ran.
#include <cassert>

bool nondet_bool();

int live = 0;
int dtors = 0;

struct C
{
  bool v;
  C(bool b) : v(b)
  {
    ++live;
  }
  ~C()
  {
    --live;
    ++dtors;
  }
  bool ok() const
  {
    return v;
  }
};

int main()
{
  int n = 0;
  while (n < 3 && C(true).ok())
    ++n;
  assert(dtors == 3);

  bool b = nondet_bool();
  if (b && C(true).ok())
    assert(dtors == 4 && live == 0);
  else
    assert(dtors == 3);
  dtors = 3;

  if (n == 3 && C(true).ok())
    assert(live == 0);

  if (C(true).ok() && live == 1 && C(true).ok() && live == 2)
    assert(live == 0);
  else
    assert(0);

  if (C(false).ok() || C(true).ok())
    assert(live == 0);

  for (int i = 0; i < 2 && C(true).ok(); ++i)
    assert(live == 0);

  int k = 0;
  do
    ++k;
  while (k < 2 ? C(true).ok() : false);
  assert(live == 0);
}
