struct b
{
  int a;
};
int main()
{
  int x = nondet_int();
  struct b g = {!x};
  2 /
    (g.a -
     2); // no division by zero (g.a can only be 0 or 1, so g.a - 2 can only be -2 or -1)
}