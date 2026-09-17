/* Regression for #7434: originfile is emitted only when a step's file differs
 * from the program file, so a single-file task's witness is unchanged and no
 * existing SV-COMP task can regress on it. */
int main(void)
{
  int d = 0;
  return 10 / d;
}
