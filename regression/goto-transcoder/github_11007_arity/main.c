void *__builtin___memcpy_chk();
int main()
{
  char a[4] = {1, 2, 3, 4}, b[4];
  __builtin___memcpy_chk(b, a);
  return 0;
}
