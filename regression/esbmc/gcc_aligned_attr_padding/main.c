struct __attribute__((aligned(16))) s
{
  int a;
  char c;
};

extern struct s g;

int main(void)
{
  return g.a;
}
