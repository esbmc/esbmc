// Single source of truth for irep2 type kinds.
//
// Includers must `#define IREP2_TYPE(kind, pretty_name)` before
// including this file and `#undef IREP2_TYPE` after. The macro is
// invoked once per kind in declaration order; the order matches the
// `type2t::type_ids` enum exactly.
//
// Arguments:
//   kind         — bare identifier (e.g. `array`). The generators
//                  derive `<kind>_type2t`, `<kind>_id`,
//                  `is_<kind>_type`, `to_<kind>_type`,
//                  `try_to_<kind>_type` from it.
//   pretty_name  — string literal printed by pretty() / used by
//                  irep2_bad_type_cast for diagnostics. Currently
//                  matches the kind identifier exactly for every
//                  entry, but is kept as a separate column so an
//                  occasional override (cf. expr_kinds.inc) can be
//                  added without touching every generator.

IREP2_TYPE(bool,       "bool")
IREP2_TYPE(empty,      "empty")
IREP2_TYPE(symbol,     "symbol")
IREP2_TYPE(struct,     "struct")
IREP2_TYPE(union,      "union")
IREP2_TYPE(code,       "code")
IREP2_TYPE(array,      "array")
IREP2_TYPE(vector,     "vector")
IREP2_TYPE(pointer,    "pointer")
IREP2_TYPE(unsignedbv, "unsignedbv")
IREP2_TYPE(signedbv,   "signedbv")
IREP2_TYPE(fixedbv,    "fixedbv")
IREP2_TYPE(floatbv,    "floatbv")
IREP2_TYPE(complex,    "complex")
IREP2_TYPE(cpp_name,   "cpp_name")
