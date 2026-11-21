struct S
{
  int a;
  int b;
};

struct S *g;

void f(void)
{
  struct S s;
  s.a = 1;
  s.b = 2;
  g = &s;
}

int main(void)
{
  f();
  return g->a;
}
