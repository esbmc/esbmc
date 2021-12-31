void fun1()
{
  int *data;
  data = (int *)__builtin_alloca(sizeof(*data));
  data[0] = 0;
}

void fun2()
{
  int *data;
  data = (int *)malloc(sizeof(*data));
  if(data != (int *)0)
  {
    data[0] = 0;
  }
  free((int *)data);
}

int main()
{
  puts("Running fun2()");
  fun2();
  puts("Done");

  puts("Running fun1()");
  fun1();
  puts("Done");

  return 0;
}
