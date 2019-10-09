#define __CRT__NO_INLINE /* Don't let mingw insert code */

#define overflow_def(type, name, intrinsic, op)                                \
  _Bool intrinsic(type);                                                       \
  _Bool __##name(type a, type b, type *res)                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    (*res) = a op b;                                                           \
    return intrinsic(a op b);                                                  \
  }

overflow_def(int, sadd_overflow, __ESBMC_overflows, +);
overflow_def(long int, saddl_overflow, __ESBMC_overflowsl, +);
overflow_def(long long int, saddll_overflow, __ESBMC_overflowsll, +);
overflow_def(unsigned int, uadd_overflow, __ESBMC_overflowsu, +);
overflow_def(unsigned long int, uaddl_overflow, __ESBMC_overflowsul, +);
overflow_def(unsigned long long int, uaddll_overflow, __ESBMC_overflowsull, +);

overflow_def(int, ssub_overflow, __ESBMC_overflows, -);
overflow_def(long int, ssubl_overflow, __ESBMC_overflowsl, -);
overflow_def(long long int, ssubll_overflow, __ESBMC_overflowsll, -);
overflow_def(unsigned int, usub_overflow, __ESBMC_overflowsu, -);
overflow_def(unsigned long int, usubl_overflow, __ESBMC_overflowsul, -);
overflow_def(unsigned long long int, usubll_overflow, __ESBMC_overflowsull, -);

overflow_def(int, smul_overflow, __ESBMC_overflows, *);
overflow_def(long int, smull_overflow, __ESBMC_overflowsl, *);
overflow_def(long long int, smulll_overflow, __ESBMC_overflowsll, *);
overflow_def(unsigned int, umul_overflow, __ESBMC_overflowsu, *);
overflow_def(unsigned long int, umull_overflow, __ESBMC_overflowsul, *);
overflow_def(unsigned long long int, umulll_overflow, __ESBMC_overflowsull, *);

#undef overflow_def