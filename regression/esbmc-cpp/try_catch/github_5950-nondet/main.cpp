int nondet_int();

// Non-STL generalisation of #5950.  A class-type throw whose destructor is
// non-trivial and dereferences `this` (writes a member), materialised in a bare
// (non-block) then-branch.  Before the fix, ~NonTrivial's destructor entry
// leaked past the `if` onto main's scope and ran on the fall-through path where
// the temporary was never constructed -> spurious "accessed expired variable
// pointer".  NonTrivial is trivially copyable (a plain int member), so the
// throw-copy introduces no real memory bug and the program must verify.
struct NonTrivial
{
  int v;
  NonTrivial() : v(0) {}
  ~NonTrivial() { v = 1; }
};

int main()
{
  int x = nondet_int();
  try
  {
    if (x)
      throw NonTrivial();
  }
  catch (NonTrivial &)
  {
  }
  return 0;
}
