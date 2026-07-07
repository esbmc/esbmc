#include <stdarg.h>

/* Correct va_start/va_arg/va_end usage, including a va_copy chain, a
 * va_list handed to a callee, and va_lists reached through pointers, must
 * stay VERIFICATION SUCCESSFUL with no extra property. */
int sum_tail(int count, va_list ap)
{
  /* va_list parameters are conservatively assumed started; the positional
   * vararg machinery cannot resolve their values across frames, so only
   * consume the list without asserting on the values. */
  int n = 0;
  for (int i = 0; i < count; i++)
  {
    va_arg(ap, int);
    n++;
  }
  return n;
}

int next(va_list *app)
{
  return va_arg(*app, int);
}

void copy_into(va_list *dst, va_list src)
{
  va_copy(*dst, src);
}

int sum_indirect(int count, ...)
{
  /* aq is started by a va_copy performed behind a pointer in a helper;
   * consuming it here must not be flagged. The values are consumed in this
   * frame, so they also resolve positionally. */
  va_list ap, aq;
  va_start(ap, count);
  copy_into(&aq, ap);
  int total = 0;
  for (int i = 0; i < count; i++)
    total += va_arg(aq, int);
  va_end(aq);
  va_end(ap);
  return total;
}

int consume_indirect(int count, ...)
{
  /* The callee reads through a va_list*; like a by-value va_list parameter
   * this is assumed started, and the values cannot be resolved across
   * frames, so only count the reads. */
  va_list ap;
  va_start(ap, count);
  int n = 0;
  for (int i = 0; i < count; i++)
  {
    next(&ap);
    n++;
  }
  va_end(ap);
  return n;
}

int sum(int count, ...)
{
  va_list ap, aq;
  va_start(ap, count);
  va_copy(aq, ap);
  int total = 0;
  for (int i = 0; i < count; i++)
    total += va_arg(aq, int);
  va_end(aq);
  va_end(ap);
  return total;
}

int consume(int count, ...)
{
  va_list ap;
  va_start(ap, count);
  int n = sum_tail(count, ap);
  va_end(ap);
  return n;
}

int main()
{
  __ESBMC_assert(sum(2, 3, 4) == 7, "va_start + va_copy chain");
  __ESBMC_assert(consume(2, 3, 4) == 2, "va_list passed to callee");
  __ESBMC_assert(sum_indirect(2, 3, 4) == 7, "va_copy behind a pointer");
  __ESBMC_assert(consume_indirect(2, 3, 4) == 2, "va_arg behind a pointer");
  return 0;
}
