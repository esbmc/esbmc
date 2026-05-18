/* Step recognition for the do-while shape. The standard counter
 * pattern: init below the bound, body increments, exits when the
 * back-edge cond becomes false. Same post-value as the for/while
 * form (i = 17) because at least one iteration happens to bring
 * i from 0 to 1, then iterations continue normally. */
int main()
{
  int i = 0;
  do {
    i++;
  } while (i < 17);
  __ESBMC_assert(i == 17, "do-while reaches exit at i = 17");
  return 0;
}
