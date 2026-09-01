int garr[4];

int main()
{
  int sarr[4];
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(garr) == 15, "global is 15 bytes");
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(sarr) == 15, "stack is 15 bytes");
  return sarr[0];
}
