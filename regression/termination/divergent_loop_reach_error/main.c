/* The first iteration calls reach_error(), which aborts, so every execution
 * terminates, but termination is not proved. */
extern int __VERIFIER_nondet_int(void);
void reach_error(void)
{
  __assert_fail("0", "main.c", 7, "reach_error");
}

int main(void)
{
  int x = 0;
  while (__VERIFIER_nondet_int())
  {
    if (x >= 0)
      reach_error();
    x++;
  }
  return 0;
}
