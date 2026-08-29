// Skipping the zero-length member must not cost the wrapper its ability to
// discharge a contract that genuinely holds (#6513).
typedef struct
{
  int len;
  int data[0];
} buf_t;

int g;

void f(buf_t *b)
{
  __ESBMC_requires(b != 0);
  __ESBMC_assigns(g);
  __ESBMC_ensures(g == 1);
  g = 1;
}

int main()
{
  return 0;
}
