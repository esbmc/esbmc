/* #6759 */
int y;

void f(int ok)
{
  if (!ok)
    y = 1;
}

int main(void)
{
  f(0);
  return 0;
}
