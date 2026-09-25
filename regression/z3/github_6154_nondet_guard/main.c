// GitHub #6154: a discarded `__builtin_reduce_add' statement inside a
// summarized callee lowers, at the raw-AST level the summarizer reads, to a
// zero-operand nondet side effect (see clang_c_adjust_expr.cpp's
// dont-care-about-missing-extensions fallback).  This exercises the
// `e.operands().empty()' guard in summary_apply_effect() that must be checked
// before op0() is dereferenced.  f cannot be summarized (the discarded call is
// rejected), so __ESBMC_forall falls back to skolemization/binder handling;
// the property below still holds because x is returned unmodified.
typedef int v4si __attribute__((vector_size(16)));

int f(int x, v4si v)
{
  __builtin_reduce_add(v);
  return x;
}

int main()
{
  v4si v = {1, 2, 3, 4};
  int y;
  __ESBMC_assert(__ESBMC_forall(&y, f(y, v) == y), "reduce-discard guard");
  return 0;
}
