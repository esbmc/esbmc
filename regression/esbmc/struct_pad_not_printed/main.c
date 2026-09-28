/* The literal must be non-zero: is_recursively_zero returns before the member
   walk reaches the padding test. `g` is extern so the dump prints the type. */
struct s
{
  char a;
  int c;
};

extern struct s g;

int main(void)
{
  struct s v = {1, 2};
  return v.c + g.a;
}
