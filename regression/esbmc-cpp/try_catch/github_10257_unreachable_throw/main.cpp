// #10257: U is thrown only in a function main never reaches.
struct U {};
struct E {};

void unused()
{
  throw U();
}

void check(int x)
{
  if (x < 0)
    throw E();
}

int main()
{
  check(1);
  return 0;
}
