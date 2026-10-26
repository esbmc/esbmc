int main()
{
  int *p;
  {
    int a;
    p = &a;
    goto L;
  }
  {
    int b;
  L:
    // a's block was left, so a is dead
    *p = 1;
  }
  return 0;
}
