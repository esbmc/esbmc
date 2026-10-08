typedef int v4 __attribute__((vector_size(16)));

int main(void)
{
  v4 x = {1, 2, 3, 4};
  __CPROVER_assert(x[2] == 4, "vector element");
  return 0;
}
