/* SV-COMP termination property: CHECK(init(main()), LTL(F end))
 * where `end` is true at exit/abort/return-from-main.
 *
 * `while (1) {}` never reaches an `end` event — no exit, no abort,
 * no return. The infinite execution violates the property →
 * VERIFICATION FAILED.
 *
 * This pins the soundness of the --termination reduction against the
 * old goto_loop_simplify behaviour, which collapsed the loop to
 * assume(false) and let FC close UNSAT at k=1, masking the
 * non-termination as SUCCESSFUL. */
int main()
{
  while (1)
  {
  }
  return 0;
}
