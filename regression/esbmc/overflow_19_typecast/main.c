int nondet_int();

int main()
{
  short a;
  int b=nondet_int();

  a = (short)b;
}
