int nondet_int(void);
int main()
{
  char buf[8];
  char *p = buf;
  if (nondet_int())
    goto check;
  p = buf + 1;
check:
  // The jump lands on the label; the size call hoisted out of the assertion
  // must still run on that path.
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(p) == 8, "size at a jump target");
  return 0;
}
