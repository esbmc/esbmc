/* goto_k_induction replaces the loop head with its havoc block before the
 * recogniser runs, so --synthesise-loop-invariants can never fire under
 * --k-induction. The run must say so rather than look like one where no loop
 * happened to be affine. */
int main(void)
{
  unsigned int i = 0, s = 0;
  while (i < 4)
  {
    s = s + 2;
    i = i + 1;
  }
  return s;
}
