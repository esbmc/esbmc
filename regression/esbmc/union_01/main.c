typedef union
{
  int a
} b;
struct c d;
struct c
{
  b a
} e(b f)
{
}
g(struct c *f)
{
  f->a.a = 2;
  e(f->a);
}
main()
{
  g(&d);
  assert(d.a.a == 1);
}
