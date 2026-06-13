#include <set>
#include <cassert>

// Condition coverage over a std::set whose insert() OM contains an internal
// function-call side effect. Under --irep2-bodies the body round-trip used to
// leave that side effect unlifted with a nil sub-field; it reached the SMT
// backend and crashed in convert_ast (crc() on a nil operand). Restoring the
// value-operand locations after the round-trip keeps goto_convert's lowering of
// the if-guards intact, so the side effect is lowered as on the legacy path.
int main()
{
  std::set<int> s;

  std::pair<std::set<int>::iterator, bool> r1 = s.insert(5);
  if (r1.second)
    s.insert(10);

  std::pair<std::set<int>::iterator, bool> r2 = s.insert(5);
  if (r2.second)
    s.insert(15);

  assert(s.size() == 2);
  return 0;
}
