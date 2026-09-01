int garr[4];

int main()
{
  int sarr[4];
  __CPROVER_assert(__CPROVER_r_ok(garr, 16), "global 16 readable");
  __CPROVER_assert(__CPROVER_r_ok(sarr, 16), "stack 16 readable");
  __CPROVER_assert(__CPROVER_w_ok(sarr, 16), "stack 16 writable");
  return sarr[0] + garr[0];
}
