// C++ [expr.cond]/2: a throw-expression operand contributes no value. A
// class-typed conditional is materialised by taking the address of each
// branch, which for the throw is meaningless -- it used to abort inside the
// solver with an "Unrecognized address_of operand" irep dump. Report it at
// the frontend instead, until the branch is lowered to a statement
// (issue #6717, from g++.dg/expr/cond16.C).
struct S
{
  const char *s;
  S(const char *p) : s(p)
  {
  }
  bool empty() const
  {
    return !s;
  }
};

S foo()
{
  S s("foo");
  return s.empty() ? throw "empty" : s;
}

int main()
{
  foo();
  return 0;
}
