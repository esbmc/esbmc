int main()
{
  int var;
  unsigned long int ptr = &var;
  int *c = (int *)ptr;
  *c = 42;
  return 0;
}
