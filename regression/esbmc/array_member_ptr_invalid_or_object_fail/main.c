typedef struct { unsigned a; unsigned arr[4]; } S;
S g;
int main(void)
{
  S *p;
  if (nondet_int())
    p = &g;
  unsigned *q = &p->arr[1];
  return *q;
}
