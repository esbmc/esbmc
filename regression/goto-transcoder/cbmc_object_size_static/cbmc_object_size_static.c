int garr[4];

int main()
{
  int sarr[4];
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(garr) == 16, "global is 16 bytes");
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(sarr) == 16, "stack is 16 bytes");
  return sarr[0];
}
