/* Step recognition for the descending counter pattern:
 * for(i=17; i>0; i--) ; should rewrite to i = 0. Exercises the
 * `step_negative` + GT branch of compute_post_value. */
int main()
{
  int i;
  for (i = 17; i > 0; i--)
    ;
  __ESBMC_assert(i == 0, "decrement step recognition");
  return 0;
}
