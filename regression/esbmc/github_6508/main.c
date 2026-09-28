struct S
{
  int buf[4];
  unsigned n;
};

int main()
{
  struct S s;
  struct S *p = &s;
  p->n = 4;
  p->buf[p->n] = 1;
  return 0;
}
