// x takes 2^32 iterations to repeat, too many for the recurrent-state check,
// but no state reaches the exit, which the inductive step shows.
unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  unsigned x = __VERIFIER_nondet_uint();
  while (1)
    x++;
  return 0;
}
