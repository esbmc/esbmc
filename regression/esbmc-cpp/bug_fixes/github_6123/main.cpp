// GitHub discussion #6123: asserts calling helper functions whose return
// value is a quantifier containing function calls. Before PR #6105 the calls
// were hoisted out of the binder, freezing the bound variable ("unique 3"
// failed spuriously).
#define SIZE 4
typedef int data_t;
typedef int idx_t;

bool fun_in_lelt(idx_t lo, idx_t primary, idx_t hi)
{
  return lo <= primary && primary < SIZE && primary < hi;
}

#define macro_in_lelt(a, b, c) (((a <= b) && (b < SIZE) && (b < c)))

bool lt_hi_1(data_t vec[SIZE], idx_t hi)
{
  idx_t a_idx;
  return __ESBMC_forall(
    &a_idx, !(0 <= a_idx) || !(a_idx < SIZE) || vec[a_idx] < hi);
}

bool lt_hi_2(data_t vec[SIZE], idx_t hi)
{
  idx_t a_idx;
  return __ESBMC_forall(
    &a_idx, !(fun_in_lelt(0, a_idx, SIZE)) || vec[a_idx] < hi);
}

bool unique_vec_1(data_t vec[SIZE], idx_t lo, idx_t hi)
{
  idx_t a_idx, b_idx;
  return __ESBMC_forall(
    &a_idx,
    !(0 <= a_idx) || !(a_idx < SIZE) ||
      __ESBMC_forall(
        &b_idx,
        !(0 <= b_idx) || !(b_idx < SIZE) || !(a_idx != b_idx) ||
          vec[a_idx] != vec[b_idx]));
}

bool unique_vec_2(data_t vec[SIZE], idx_t lo, idx_t hi)
{
  idx_t a_idx, b_idx;
  return __ESBMC_forall(
    &a_idx,
    !(macro_in_lelt(0, a_idx, hi)) ||
      __ESBMC_forall(
        &b_idx,
        !(macro_in_lelt(0, b_idx, hi)) || !(a_idx != b_idx) ||
          vec[a_idx] != vec[b_idx]));
}

bool unique_vec_3(data_t vec[SIZE], idx_t lo, idx_t hi)
{
  idx_t a_idx, b_idx;
  return __ESBMC_forall(
    &a_idx,
    !(fun_in_lelt(0, a_idx, hi)) ||
      __ESBMC_forall(
        &b_idx,
        !(fun_in_lelt(0, b_idx, hi)) || !(a_idx != b_idx) ||
          vec[a_idx] != vec[b_idx]));
}

int main()
{
  data_t vec[SIZE];
  vec[0] = 0;
  vec[1] = 1;
  vec[2] = 2;
  vec[3] = 3;

  idx_t a_idx, b_idx;
  idx_t lo = 0;
  idx_t hi = SIZE;

  __ESBMC_assert(lt_hi_1(vec, hi), "lt 1");
  __ESBMC_assert(lt_hi_2(vec, hi), "lt 2");

  __ESBMC_assert(
    __ESBMC_forall(
      &a_idx,
      !(0 <= a_idx) || !(a_idx < SIZE) ||
        __ESBMC_forall(
          &b_idx,
          !(0 <= b_idx) || !(b_idx < SIZE) || !(a_idx != b_idx) ||
            vec[a_idx] != vec[b_idx])),
    "unique 0");

  __ESBMC_assert(unique_vec_1(vec, 0, SIZE), "unique 1");
  __ESBMC_assert(unique_vec_2(vec, 0, SIZE), "unique 2");
  __ESBMC_assert(unique_vec_3(vec, 0, SIZE), "unique 3");
}
