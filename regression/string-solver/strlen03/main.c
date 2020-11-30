int main()
{
  int i;
  char strA[7];
  strcpy(strA, "012");
  i = strlen(strA);
  assert(i == 3);
  assert(strA[1] == '1');
  strcpy(strA, "4567");
  i = strlen(strA);
  assert(i == 4);
  assert(strA[1] == '5');
  return 0;
}
