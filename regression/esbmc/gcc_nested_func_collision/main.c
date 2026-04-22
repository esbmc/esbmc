// Regression test for #4098: name collision with underscore-heavy identifiers.
// two_fun__ + __local and two_fun____ + local must produce distinct lifted names.
void two_fun__(void)
{
  int x = 0;
  void __local(void)
  {
    x++;
  }
  __local();
  __local();
  if (x <= 2)
    __local();
  __ESBMC_assert(x == 3, "two_fun__ nested calls");
}

void two_fun____(void)
{
  int x = 0;
  void local(void)
  {
    x++;
  }
  local();
  local();
  if (x <= 2)
    local();
  __ESBMC_assert(x == 3, "two_fun____ nested calls");
}

int main(void)
{
  two_fun__();
  two_fun____();
  return 0;
}
