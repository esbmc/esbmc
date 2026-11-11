// #10257: U is thrown in a function main calls.
struct U {};
struct E {};

void used()
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
  used();
  return 0;
}
