#include <irep2/irep2_instantiate.h>
#include <irep2/irep2_type.h>

ESBMC_INSTANTIATE_TYPE_EMPTY(bool_type, type2t);
ESBMC_INSTANTIATE_TYPE_EMPTY(empty_type, type2t);
ESBMC_INSTANTIATE_TYPE(symbol_type, symbol_type_data);
ESBMC_INSTANTIATE_TYPE(struct_type, struct_union_data);
ESBMC_INSTANTIATE_TYPE(union_type, struct_union_data);
ESBMC_INSTANTIATE_TYPE(unsignedbv_type, bv_data);
ESBMC_INSTANTIATE_TYPE(signedbv_type, bv_data);
ESBMC_INSTANTIATE_TYPE(code_type, code_data);
ESBMC_INSTANTIATE_TYPE(array_type, array_data);
ESBMC_INSTANTIATE_TYPE(vector_type, array_data);
ESBMC_INSTANTIATE_TYPE(pointer_type, pointer_data);
ESBMC_INSTANTIATE_TYPE(fixedbv_type, fixedbv_data);
ESBMC_INSTANTIATE_TYPE(floatbv_type, floatbv_data);
ESBMC_INSTANTIATE_TYPE(complex_type, struct_union_data);
ESBMC_INSTANTIATE_TYPE(cpp_name_type, cpp_name_data);
