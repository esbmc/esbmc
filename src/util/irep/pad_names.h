#ifndef ESBMC_UTIL_IREP_PAD_NAMES_H
#define ESBMC_UTIL_IREP_PAD_NAMES_H

#include <string_view>
#include <util/base/prefix.h>

/* Names of the synthetic members add_padding() inserts into struct/union types.
 * They contain '#', which no C or C++ identifier may contain -- unlike '$',
 * which clang accepts in identifiers by default -- so a declared member can
 * never collide with one (esbmc/esbmc#1476). Collisions are not merely
 * cosmetic: struct_union_get_component_number() resolves a member by name and
 * yields nothing when the name is ambiguous. */
inline constexpr std::string_view pad_prefix = "anon_pad#";
inline constexpr std::string_view pad_bit_field_prefix = "anon_bit_field_pad#";
inline constexpr std::string_view pad_ext_int_prefix = "ext_int_pad#";
inline constexpr std::string_view pad_union_name = "union_pad#";

inline bool is_padding_name(std::string_view name)
{
  return has_prefix(name, pad_prefix) ||
         has_prefix(name, pad_bit_field_prefix) ||
         has_prefix(name, pad_ext_int_prefix) || name == pad_union_name;
}

#endif
