/* The witness path reaches goto_loop_invariant_combined too, so the same
 * exemption applies: a witness-injected loop invariant does not put an ASSERT
 * in the loop head's slot, and --termination's havoc composes with it. */
unsigned int nondet_uint(void);

int main(void)
{
  unsigned int i = 0;
  unsigned int n = nondet_uint();
  unsigned int s = 0;

  while (i != n)
  {
    s = s + 1;
    i = i + 1;
  }

  return 0;
}
