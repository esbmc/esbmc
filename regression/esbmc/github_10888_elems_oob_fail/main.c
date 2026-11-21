/* #10888: a struct spanning past the last element is still out of bounds. */
struct U
{
  unsigned char f1, f2;
};
struct G
{
  struct U f1, f2;
};
unsigned char nondet_uchar(void);
int main(void)
{
  struct U arr[2] = {
    {nondet_uchar(), nondet_uchar()}, {nondet_uchar(), nondet_uchar()}};
  struct G g = *(struct G *)&arr[1];
  return g.f1.f1;
}
