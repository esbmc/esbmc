enum AB { A, B };

int (x);

void func(enum AB foo)
{
  assert(foo == A);
}

int main()
{
  enum BA { A1=-10, B1=33, C1 };
  enum BA foo1 = 4;
  func(A);
  return 0;
}
