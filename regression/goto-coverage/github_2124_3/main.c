void sort(int *ptr)
{
  if (ptr[1] == ptr[2])
    ;
}

int main()
{
  int x = 1;
  int *y = &x;
  sort(y);
}