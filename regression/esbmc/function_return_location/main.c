int Foo(int F_a, int F_b)
{
  int F_res = F_a + F_b;
  return F_res;
}

int main()
{
  int M_x, M_y, M_z;
  M_z = Foo(M_x, M_y);
  __ESBMC_assert(M_z == M_x + M_y + 1, "result");
}
