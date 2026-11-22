// The loop writes a bit-field, a struct member array and an array, and the
// state comes back after two iterations. The inductive step would show this
// too, so the expected message pins the recurrent-state check.
struct s
{
  unsigned c : 2;
  int a[2];
};

int main(void)
{
  struct s v = {0, {0, 1}};
  int b[2] = {1, 0};
  while (v.c != 2)
  {
    v.c ^= 1;
    v.a[v.c] = v.c;
    b[v.c] = !v.c;
  }
  return 0;
}
