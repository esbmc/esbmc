struct b
{
  int a;
};
int main()
{
  int x = nondet_int();
  struct b g = {!x};
  2 / (g.a - 1); // division by zero (g.a can be 1)
}