/* do empty ones */
struct I_AM_EMPTY
{
};

struct x {
  int f;
  int g;
};

struct y {
  struct x X;
  int h;
};

int main()
{
  struct y Y;

  {
    struct x tmp;
    tmp = Y.X;
    tmp.f = 1;
    Y.X = tmp;
  }

  assert(Y.X.f != 1);
}
