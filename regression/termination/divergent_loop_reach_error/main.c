/* The first iteration calls reach_error(), which aborts, so every execution
 * terminates. --no-assertions empties reach_error() in GOTO; the loop must
 * still not be taken for one that can run forever. */
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
