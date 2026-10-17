// esbmc/esbmc#8681: the operand of delete is checked like any expression.
int nondet_int();
int main()
{
  int *a[2] = {nullptr, nullptr};
  int k = nondet_int();
  delete a[1 / k];
  return 0;
}
