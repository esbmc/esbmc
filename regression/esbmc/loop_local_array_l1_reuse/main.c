int nondet_int();
int main()
{
  int x = nondet_int();
  int r = 0;
  while (nondet_int())
  {
    int t[2] = {x, 0};
    r = t[1];
  }
  return r;
}
