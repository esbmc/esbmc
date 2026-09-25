/* The overflow claim the promotion keeps falsifiable: 3 + INT_MAX overflows in
   int. Narrowing `a` to char first would make it unreachable. */
char b;

int main(void)
{
  int a = 2147483647;

  b = 3;
  b += a;
  return 0;
}
