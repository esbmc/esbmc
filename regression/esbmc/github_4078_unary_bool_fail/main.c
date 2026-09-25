// Negative counterpart of github_4078_unary_bool: the promoted operand
// carries its real value, so a claim contradicting it is refuted rather than
// vacuously held (issue #4078).
int a, b;

int main(void)
{
  __ESBMC_assert(~(a || b) == 0, "must not hold");
  return 0;
}
