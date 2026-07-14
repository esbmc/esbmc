extern void abort(void);
extern void __assert_fail(const char *, const char *, unsigned int, const char *)
  __attribute__((__nothrow__, __leaf__)) __attribute__((__noreturn__));

void reach_error()
{
  __assert_fail("0", "test_locks_13.c", 3, "reach_error");
}

extern int __VERIFIER_nondet_int();

int main()
{
  int p1 = __VERIFIER_nondet_int();
  int lk1;
  int p2 = __VERIFIER_nondet_int();
  int lk2;
  int p3 = __VERIFIER_nondet_int();
  int lk3;
  int p4 = __VERIFIER_nondet_int();
  int lk4;
  int p5 = __VERIFIER_nondet_int();
  int lk5;
  int p6 = __VERIFIER_nondet_int();
  int lk6;
  int p7 = __VERIFIER_nondet_int();
  int lk7;
  int p8 = __VERIFIER_nondet_int();
  int lk8;
  int p9 = __VERIFIER_nondet_int();
  int lk9;
  int p10 = __VERIFIER_nondet_int();
  int lk10;
  int p11 = __VERIFIER_nondet_int();
  int lk11;
  int p12 = __VERIFIER_nondet_int();
  int lk12;
  int p13 = __VERIFIER_nondet_int();
  int lk13;
  int cond;

  while (1)
  {
    cond = __VERIFIER_nondet_int();
    if (cond == 0)
      goto out;

    lk1 = 0;
    lk2 = 0;
    lk3 = 0;
    lk4 = 0;
    lk5 = 0;
    lk6 = 0;
    lk7 = 0;
    lk8 = 0;
    lk9 = 0;
    lk10 = 0;
    lk11 = 0;
    lk12 = 0;
    lk13 = 0;

    if (p1 != 0)
      lk1 = 1;
    if (p2 != 0)
      lk2 = 1;
    if (p3 != 0)
      lk3 = 1;
    if (p4 != 0)
      lk4 = 1;
    if (p5 != 0)
      lk5 = 1;
    if (p6 != 0)
      lk6 = 1;
    if (p7 != 0)
      lk7 = 1;
    if (p8 != 0)
      lk8 = 1;
    if (p9 != 0)
      lk9 = 1;
    if (p10 != 0)
      lk10 = 1;
    if (p11 != 0)
      lk11 = 1;
    if (p12 != 0)
      lk12 = 1;
    if (p13 != 0)
      lk13 = 1;

    if (p1 != 0)
    {
      if (lk1 != 1)
        goto ERROR;
      lk1 = 0;
    }
    if (p2 != 0)
    {
      if (lk2 != 1)
        goto ERROR;
      lk2 = 0;
    }
    if (p3 != 0)
    {
      if (lk3 != 1)
        goto ERROR;
      lk3 = 0;
    }
    if (p4 != 0)
    {
      if (lk4 != 1)
        goto ERROR;
      lk4 = 0;
    }
    if (p5 != 0)
    {
      if (lk5 != 1)
        goto ERROR;
      lk5 = 0;
    }
    if (p6 != 0)
    {
      if (lk6 != 1)
        goto ERROR;
      lk6 = 0;
    }
    if (p7 != 0)
    {
      if (lk7 != 1)
        goto ERROR;
      lk7 = 0;
    }
    if (p8 != 0)
    {
      if (lk8 != 1)
        goto ERROR;
      lk8 = 0;
    }
    if (p9 != 0)
    {
      if (lk9 != 1)
        goto ERROR;
      lk9 = 0;
    }
    if (p10 != 0)
    {
      if (lk10 != 1)
        goto ERROR;
      lk10 = 0;
    }
    if (p11 != 0)
    {
      if (lk11 != 1)
        goto ERROR;
      lk11 = 0;
    }
    if (p12 != 0)
    {
      if (lk12 != 1)
        goto ERROR;
      lk12 = 0;
    }
    if (p13 != 0)
    {
      if (lk13 != 1)
        goto ERROR;
      lk13 = 0;
    }
  }

out:
  return 0;

ERROR:
  reach_error();
  abort();
  return 0;
}
