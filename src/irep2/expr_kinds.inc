// Single source of truth for irep2 expression kinds.
//
// Includers must `#define IREP2_EXPR(kind, pretty_name)` before
// including this file and `#undef IREP2_EXPR` after. The macro is
// invoked once per kind in declaration order; the order matches the
// `expr2t::expr_ids` enum exactly. Adding a new expression kind is
// a single line here.
//
// Arguments:
//   kind         — bare identifier (e.g. `add`). The generators
//                  derive `<kind>2t`, `<kind>_id`, `is_<kind>2t`,
//                  `to_<kind>2t`, `try_to_<kind>2t` from it.
//   pretty_name  — string literal printed by pretty() / used by
//                  irep2_bad_expr_cast for diagnostics. Most match
//                  `#kind` exactly, but a few are intentionally
//                  reshaped (e.g. `null_object` prints as
//                  "NULL-object", `code_cpp_*` print without the
//                  `code_` prefix).

IREP2_EXPR(constant_int,          "constant_int")
IREP2_EXPR(constant_fixedbv,      "constant_fixedbv")
IREP2_EXPR(constant_floatbv,      "constant_floatbv")
IREP2_EXPR(constant_bool,         "constant_bool")
IREP2_EXPR(constant_string,       "constant_string")
IREP2_EXPR(constant_struct,       "constant_struct")
IREP2_EXPR(constant_union,        "constant_union")
IREP2_EXPR(constant_array,        "constant_array")
IREP2_EXPR(constant_vector,       "constant_vector")
IREP2_EXPR(constant_array_of,     "constant_array_of")
IREP2_EXPR(symbol,                "symbol")
IREP2_EXPR(typecast,              "typecast")
IREP2_EXPR(bitcast,               "bitcast")
IREP2_EXPR(nearbyint,             "nearbyint")
IREP2_EXPR(if,                    "if")
IREP2_EXPR(equality,              "equality")
IREP2_EXPR(notequal,              "notequal")
IREP2_EXPR(lessthan,              "lessthan")
IREP2_EXPR(greaterthan,           "greaterthan")
IREP2_EXPR(lessthanequal,         "lessthanequal")
IREP2_EXPR(greaterthanequal,      "greaterthanequal")
IREP2_EXPR(cmp_three_way,         "cmp_three_way")
IREP2_EXPR(not,                   "not")
IREP2_EXPR(and,                   "and")
IREP2_EXPR(or,                    "or")
IREP2_EXPR(xor,                   "xor")
IREP2_EXPR(implies,               "implies")
IREP2_EXPR(bitand,                "bitand")
IREP2_EXPR(bitor,                 "bitor")
IREP2_EXPR(bitxor,                "bitxor")
IREP2_EXPR(bitnot,                "bitnot")
IREP2_EXPR(lshr,                  "lshr")
IREP2_EXPR(neg,                   "neg")
IREP2_EXPR(abs,                   "abs")
IREP2_EXPR(add,                   "add")
IREP2_EXPR(sub,                   "sub")
IREP2_EXPR(mul,                   "mul")
IREP2_EXPR(div,                   "div")
IREP2_EXPR(ieee_add,              "ieee_add")
IREP2_EXPR(ieee_sub,              "ieee_sub")
IREP2_EXPR(ieee_mul,              "ieee_mul")
IREP2_EXPR(ieee_div,              "ieee_div")
IREP2_EXPR(ieee_fma,              "ieee_fma")
IREP2_EXPR(ieee_sqrt,             "ieee_sqrt")
IREP2_EXPR(popcount,              "popcount")
IREP2_EXPR(bswap,                 "bswap")
IREP2_EXPR(modulus,               "modulus")
IREP2_EXPR(shl,                   "shl")
IREP2_EXPR(ashr,                  "ashr")
IREP2_EXPR(dynamic_object,        "dynamic_object")
IREP2_EXPR(same_object,           "same_object")
IREP2_EXPR(pointer_offset,        "pointer_offset")
IREP2_EXPR(pointer_object,        "pointer_object")
IREP2_EXPR(pointer_capability,    "pointer_capability")
IREP2_EXPR(address_of,            "address_of")
IREP2_EXPR(byte_extract,          "byte_extract")
IREP2_EXPR(byte_update,           "byte_update")
IREP2_EXPR(with,                  "with")
IREP2_EXPR(member,                "member")
IREP2_EXPR(member_ref,            "member_ref")
IREP2_EXPR(ptr_mem,               "ptr_mem")
IREP2_EXPR(index,                 "index")
IREP2_EXPR(isnan,                 "isnan")
IREP2_EXPR(overflow,              "overflow")
IREP2_EXPR(overflow_cast,         "overflow_cast")
IREP2_EXPR(overflow_neg,          "overflow_neg")
IREP2_EXPR(unknown,               "unknown")
IREP2_EXPR(invalid,               "invalid")
IREP2_EXPR(null_object,           "NULL-object")
IREP2_EXPR(dereference,           "dereference")
IREP2_EXPR(valid_object,          "valid_object")
IREP2_EXPR(races_check,           "races_check")
IREP2_EXPR(deallocated_obj,       "deallocated_obj")
IREP2_EXPR(dynamic_size,          "dynamic_size")
IREP2_EXPR(sideeffect,            "sideeffect")
IREP2_EXPR(code_block,            "code_block")
IREP2_EXPR(code_assign,           "code_assign")
IREP2_EXPR(code_decl,             "code_decl")
IREP2_EXPR(code_dead,             "code_dead")
IREP2_EXPR(code_printf,           "code_printf")
IREP2_EXPR(code_expression,       "code_expression")
IREP2_EXPR(code_return,           "code_return")
IREP2_EXPR(code_skip,             "code_skip")
IREP2_EXPR(code_free,             "code_free")
IREP2_EXPR(code_goto,             "code_goto")
IREP2_EXPR(object_descriptor,     "object_descriptor")
IREP2_EXPR(code_function_call,    "code_function_call")
IREP2_EXPR(code_comma,            "code_comma_id")
IREP2_EXPR(invalid_pointer,       "invalid_pointer")
IREP2_EXPR(code_asm,              "code_asm")
IREP2_EXPR(code_cpp_del_array,    "cpp_del_array")
IREP2_EXPR(code_cpp_delete,       "cpp_delete")
IREP2_EXPR(code_cpp_catch,        "cpp_catch")
IREP2_EXPR(code_cpp_throw,        "cpp_throw")
IREP2_EXPR(isinf,                 "isinf")
IREP2_EXPR(isnormal,              "isnormal")
IREP2_EXPR(isfinite,              "isfinite")
IREP2_EXPR(signbit,               "signbit")
IREP2_EXPR(concat,                "concat")
IREP2_EXPR(extract,               "extract")
IREP2_EXPR(capability_base,       "capability_base")
IREP2_EXPR(capability_top,        "capability_top")
IREP2_EXPR(forall,                "forall")
IREP2_EXPR(exists,                "exists")
IREP2_EXPR(isinstance,            "isinstance")
IREP2_EXPR(hasattr,               "hasattr")
IREP2_EXPR(isnone,                "isnone")
// V.4 structured control-flow code kinds (esbmc/esbmc#4715). Appended at the
// end so existing expr_id enum values are not renumbered.
IREP2_EXPR(code_ifthenelse,       "code_ifthenelse")
IREP2_EXPR(code_while,            "code_while")
IREP2_EXPR(code_dowhile,          "code_dowhile")
IREP2_EXPR(code_for,              "code_for")
IREP2_EXPR(code_switch,           "code_switch")
IREP2_EXPR(code_break,            "code_break")
IREP2_EXPR(code_continue,         "code_continue")
IREP2_EXPR(code_label,            "code_label")
IREP2_EXPR(code_switch_case,      "code_switch_case")
IREP2_EXPR(code_assert,           "code_assert")
IREP2_EXPR(code_assume,           "code_assume")
// Assignment side-effects (sideeffect("assign"), "assign+", etc.) used in the
// C frontend as expression-level assignments and compound-assignment operators.
// Encoded as a single IREP2 kind carrying the operator string so that every
// variant round-trips exactly without requiring one enum entry per operator.
IREP2_EXPR(sideeffect_assign,     "sideeffect_assign")
// C++ constructor "this" placeholder: exprt("new_object") in the legacy IR.
// Appears in temporary_object initializers; replaced by replace_new_object
// before goto_convert processes the body.
IREP2_EXPR(new_object,            "new_object")
// sizeof(T) carrying the measured type T as a first-class reflected field,
// replacing the legacy sizeof-type irep attribute side channel
// (esbmc/esbmc#5337). Folds lazily to the byte size of T via do_simplify().
IREP2_EXPR(sizeof,                "sizeof")
// Uninterpreted-function application. Created during symbolic execution to
// model a call to a "__ESBMC_uninterpreted_*" / "__CPROVER_uninterpreted_*"
// function, and consumed only at SMT conversion time, where the solver's
// native uninterpreted-function support enforces functional congruence.
IREP2_EXPR(uninterpreted_func,    "uninterpreted_func")
