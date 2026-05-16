#include <irep2/irep2_instantiate.h>
#include <irep2/irep2_expr.h>

ESBMC_INSTANTIATE_EXPR(not, bool_1op);
ESBMC_INSTANTIATE_EXPR(ieee_fma, ieee_arith_3ops);
ESBMC_INSTANTIATE_EXPR(ieee_sqrt, ieee_arith_1op);
ESBMC_INSTANTIATE_EXPR(pointer_offset, pointer_ops);
ESBMC_INSTANTIATE_EXPR(address_of, pointer_ops);
ESBMC_INSTANTIATE_EXPR(overflow, overflow_ops);
ESBMC_INSTANTIATE_EXPR(invalid_pointer, invalid_pointer_ops);
ESBMC_INSTANTIATE_EXPR(concat, bit_2ops);
ESBMC_INSTANTIATE_EXPR(overflow_neg, overflow_ops);
