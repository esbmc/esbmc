/* Binding a nondet read out of a row to a variable puts an index over a
 * row-valued `with` in front of smt_solver_baset::get() when the trace is
 * built. get()'s index_id case mirrors convert_array_index()'s dispatch, so it
 * needs the same row lowering: without it convert_ast() is handed a row, which
 * has no term, and ESBMC aborts on sort->id == SMT_SORT_ARRAY instead of
 * printing the counterexample master prints. */
int nondet_int(void);

int main(void)
{
  int a[2][2];
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);

  a[1][0] = 5;
  a[1][1] = 4;

  int v = a[1][i];
  __ESBMC_assert(v == 5, "the trace must name the element the row read reached");
  return 0;
}
