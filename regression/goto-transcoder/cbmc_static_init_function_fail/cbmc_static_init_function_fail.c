/* An extern CBMC never saw defined is nondet, not zero. */
extern int unlinked;

void harness(void)
{
  __CPROVER_assert(unlinked == 0, "unlinked extern is nondet");
}

int main(void)
{
  return 0;
}
