struct S { int a[2]; };
int n;
struct S f() { struct S s = {{n > 1, 0}}; n++; return s; }
int main()
{
  int *q = 0;
  while (1)
    if ((q = f().a)[0])
      break;
  return q[1];
}
