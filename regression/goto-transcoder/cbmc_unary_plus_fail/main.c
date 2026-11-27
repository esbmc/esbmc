/* Unary plus does not change its operand's value. */
int main(void)
{
  int i = 3;
  __CPROVER_assert(+i == 4, "+i is 3, not 4");
  return 0;
}
