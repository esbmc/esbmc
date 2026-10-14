typedef float v4f __attribute__((vector_size(16)));
int main(void)
{
  v4f a = {1.5f, 2.5f, 3.5f, 4.5f};
  v4f b = {0.5f, 0.5f, 0.5f, 0.5f};
  v4f s = a + b;
  v4f d = a - b;
  v4f m = a * b;
  v4f q = a / b;
  __ESBMC_assert(s[0] == 2.0f && s[3] == 5.0f, "vector add");
  __ESBMC_assert(d[0] == 1.0f, "vector sub");
  __ESBMC_assert(m[1] == 1.25f, "vector mul");
  __ESBMC_assert(q[0] == 3.0f, "vector div");
  return 0;
}
