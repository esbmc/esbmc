int f(void) { return 0; }
int main()
{
  *(int *)(void *)f = 1;
  return 0;
}
