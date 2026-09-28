volatile int v;
int main()
{
  int a = v;
  int b = v;
  __CPROVER_assert(a == b, "two reads of a volatile agree");
  return 0;
}
