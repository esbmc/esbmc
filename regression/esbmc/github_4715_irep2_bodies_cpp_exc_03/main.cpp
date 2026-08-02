// Exercises throwing a CLASS TEMPORARY under --irep2-bodies (V.4.3, esbmc#4715).
// `throw A(42)` builds a temporary_object side effect whose initializer is the
// constructor call. convert_throw was only reached via the side_effect_exprt
// path in remove_sideeffects (which lowers operands first); the --irep2-bodies
// body round-trip turns the throw into a codet("cpp-throw") dispatched straight
// to convert_throw, so the temporary_object operand reached the exception
// machinery un-lowered — the constructor never ran and the handler read an
// unconstructed object (spurious FAILED on `a.x == 42`). convert_throw now lowers
// its operand's side effects, so the constructor runs and the assertion holds.
struct A
{
  int x;
  A(int v) : x(v)
  {
  }
};

int main()
{
  try
  {
    throw A(42);
  }
  catch (A &a)
  {
    __ESBMC_assert(a.x == 42, "thrown object was constructed");
    return 1;
  }
  return 0;
}
