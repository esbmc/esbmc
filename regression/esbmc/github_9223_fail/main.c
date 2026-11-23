int nondet_int(void);
int main(void)
{
  int a = nondet_int();
  int b = nondet_int();
  return a << (b & 15);
}
