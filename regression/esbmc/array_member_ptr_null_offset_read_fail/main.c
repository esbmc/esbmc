typedef struct { unsigned a; unsigned arr[4]; unsigned z; } S;
S g;
int main(void)
{
  S *p;
  if (nondet_int()) p = &g; else p = (S *)((char *)0 + 16);
  unsigned *q = &p->arr[1];
  unsigned v = *q;
  return v;
}
