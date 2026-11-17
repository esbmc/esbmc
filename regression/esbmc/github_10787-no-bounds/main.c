// &b.a[2] points past the one-past-the-end address of b.a (C11 6.5.6p8).
struct
{
  int a[1];
  int end;
} b;
int main()
{
  int c = 2;
  int *p = &b.a[c];
  return 0;
}
