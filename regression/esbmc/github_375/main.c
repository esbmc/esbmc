#include <stdio.h>
#include <string.h>

#define _4GB 10000
#define _2GB 5000
#define _32MB 1000
#define _16MB 500
#define _2MB 100

typedef struct _RANGE
{
  unsigned int len;
  unsigned int base;
  unsigned int top;
} RANGE;

unsigned int nondet_ull()
{
  unsigned int val;
  return val;
}

typedef union N_RANGE
{
  RANGE Raw[4];
  struct
  {
    RANGE SegArr[2];
    RANGE ObjArr[2];
  };
} N_RANGE;

N_RANGE RangeArr;

void nondet_RANGE_by_fields(RANGE *val)
{
  val->len = nondet_ull();
  val->base = nondet_ull();
  val->top = val->len + val->base;
}

RANGE nondet_RANGE()
{
  RANGE val;

  val.len = nondet_ull();
  val.base = nondet_ull();
  val.top = val.len + val.base;

  return val;
}

#define nondet_RANGE_fields(some_range)                                        \
  ({                                                                           \
    (some_range).base = nondet_ull();                                          \
    (some_range).len = nondet_ull();                                           \
    (some_range).top = (some_range).base + (some_range).len;                   \
  })

void range_fun()
{
  unsigned int i = 0;
  unsigned int sum_rng_len = 0;
  unsigned int highest_range_top = _4GB - _16MB;
  unsigned int current_base;
  unsigned int current_len;

  for(; i < 4; i++)
  {
    RangeArr.Raw[i] = nondet_RANGE();
    //nondet_RANGE_by_fiedls(&RangeArr.Raw[i]);
    nondet_RANGE_fields(RangeArr.Raw[i]);
    __ESBMC_assume(RangeArr.Raw[i].base >= _4GB - _16MB);

    current_base = RangeArr.Raw[i].base;
    current_len = RangeArr.Raw[i].len;
    sum_rng_len += RangeArr.Raw[i].len;
    if(i < 2)
      if(RangeArr.Raw[i].top > highest_range_top)
        highest_range_top = RangeArr.Raw[i].top;

    //__ESBMC_assert(RangeArr.Raw[i].len <= _16MB, "RangeArr len must be less than _16MB");
  }

  //__ESBMC_assume(sum_rng_len <= _16MB);
  //__ESBMC_assume(highest_range_top == _4GB);
  //__ESBMC_assert(sum_rng_len != _2MB, "sum_rng_len should be less than _16MB");
  __ESBMC_assert(RangeArr.Raw[0].base == _2GB, "RangeArr base value 1");
  //__ESBMC_assert(RangeArr.Raw[0].len == _2MB, "RangeArr len value 1");
}

int main()
{
  range_fun();
  return 0;
}
