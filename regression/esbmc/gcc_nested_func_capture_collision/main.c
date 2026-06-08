// Regression: capture param name must not collide with user's __capture_x.
int main(void)
{
  int x = 10;
  int result(void)
  {
    int __capture_x = 42;
    x += __capture_x;
    return x;
  }
  int r = result();
  __ESBMC_assert(r == 52, "capture collision");
  __ESBMC_assert(x == 52, "captured x updated");
  return 0;
}
