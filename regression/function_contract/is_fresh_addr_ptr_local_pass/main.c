typedef struct
{
  int coeffs[4];
} P;

void f(void)
{
  P *v;
  __ESBMC_requires(__ESBMC_is_fresh(&v, sizeof(P)));
}

int main(void)
{
  return 0;
}
