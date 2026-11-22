/* A CBMC goto-binary without a __KANI_ symbol keeps plain BMC. */
unsigned nondet_uint(void);
int main(void)
{
  unsigned x = nondet_uint() % 4;
  __CPROVER_assert(x < 4, "x stays below 4");
  return 0;
}
