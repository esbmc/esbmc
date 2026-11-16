unsigned nondet_uint(void);
int main(void)
{
  unsigned a = nondet_uint(), b = nondet_uint();
  return a + b;
}
