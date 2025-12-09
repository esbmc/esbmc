#include <irep2/irep2_templates_types.h>

type_typedefs_empty(bool_type, type2t);
type_typedefs_empty(empty_type, type2t);
type_typedefs1(symbol_type, symbol_type_data);
type_typedefs5(struct_type, struct_union_data);
type_typedefs5(union_type, struct_union_data);
type_typedefs1(unsignedbv_type, bv_data);
type_typedefs1(signedbv_type, bv_data);
type_typedefs4(code_type, code_data);
type_typedefs3(array_type, array_data);
type_typedefs3(vector_type, array_data);
type_typedefs1(pointer_type, pointer_data);
type_typedefs2(fixedbv_type, fixedbv_data);
type_typedefs2(floatbv_type, floatbv_data);
type_typedefs2(cpp_name_type, cpp_name_data);
