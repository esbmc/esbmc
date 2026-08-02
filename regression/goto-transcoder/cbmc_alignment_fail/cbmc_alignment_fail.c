/* Negative: alignment is a real constraint, not a vacuous pass. An int object
   is 4-aligned but not necessarily at address 0, so &x % 4 == 1 is false. */
int main(void)
{
  int x;
  __CPROVER_assert((unsigned long)&x % 4 == 1, "misaligned by 1 is impossible");
  return 0;
}
