enum AB { A, B };

int (x);

void func(enum AB foo)
{
  assert(foo == 0);
}

int main()
{
  enum BA { A1=-10, B1=33, C1 };
  enum BA foo1 = 4;

  enum  { A2=0, B2=33, C2 };
  func(A2);

  func(A);
  return 0;
}
