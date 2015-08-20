int nondet_int();

int Test = nondet_int();

int f()
{
  return 1;
}

int g=f();

int main()
{
  assert(g==1);
}
