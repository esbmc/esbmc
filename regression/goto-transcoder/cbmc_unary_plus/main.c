/* Unary plus yields its promoted operand. */
int main(void)
{
  int i = 3;
  char c = -1;
  __CPROVER_assert(+i == 3, "int operand");
  __CPROVER_assert(+c == -1 && sizeof(+c) == sizeof(int), "char is promoted");
  return 0;
}
