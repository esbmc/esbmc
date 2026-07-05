// Regression for the --ir (integer/real) fixedbv encoding: a fixedbv->int
// cast forces round_real_to_int, which requires a REAL operand. A
// bitcast<fixedbv>(bitvector) used to be left int-sorted under --ir, so the
// operand reached mk_lt with mismatched sorts and aborted
// (z3_conv.cpp mk_lt: a->sort->id == b->sort->id). The truncation toward
// zero of a constant float must hold.
int main()
{
  float x = 2.5f;
  int n = (int)x;
  __ESBMC_assert(n == 2, "C casts truncate toward zero");
  return 0;
}
