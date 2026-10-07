typedef int v4 __attribute__((vector_size(16)));
struct with_vector
{
  int a;
  v4 v;
};

int main(void)
{
  v4 x = {1, 2, 3, 4};
  struct with_vector s = {5, {6, 7, 8, 9}};
  __CPROVER_assert(x[2] == 3 && s.a == 5 && s.v[3] == 9, "vector elements");
  return 0;
}
