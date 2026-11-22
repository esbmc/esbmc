/* #10888: a struct wider than its object is still out of bounds. */
struct G
{
  unsigned char f1[2], f2[2];
};
struct H
{
  unsigned char a[2];
};
unsigned char nondet_uchar(void);
int main(void)
{
  struct H h = {{nondet_uchar(), nondet_uchar()}};
  struct G g = *(struct G *)&h;
  return g.f2[0];
}
