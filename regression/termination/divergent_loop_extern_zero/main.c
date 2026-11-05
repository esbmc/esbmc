/* enabled is defined nowhere. The divergence check takes it to be 0, not any
 * value, so the loop is not reported as reachable. */
extern int __VERIFIER_nondet_int(void);
extern int enabled;

int main(void)
{
  if (!enabled)
    return 0;
  while (__VERIFIER_nondet_int())
    ;
  return 0;
}
