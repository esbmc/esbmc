#ifndef UTIL_SYMTAB_BASE_SUBOBJECT_H
#define UTIL_SYMTAB_BASE_SUBOBJECT_H

#include <string>
#include <string_view>

// Prefix of the nested base-subobject components the C++ frontend adds to a
// derived struct, one per direct base. Inherited member access, up/downcasts
// and base ctor/dtor `this` are all routed through these components rather
// than through clang-ABI byte offsets. Shared with goto-programs and
// goto-symex, which decode the same layout. See esbmc/esbmc#1866, #3894.
inline constexpr std::string_view BASE_SUBOBJECT_PREFIX = "@base@";

// Component name for a direct base whose class_id is `class_id`. Must agree
// between the storage site (get_base_components_methods), the cast handlers
// and the destructor chain.
inline std::string base_subobject_name(std::string_view class_id)
{
  return std::string(BASE_SUBOBJECT_PREFIX) + std::string(class_id);
}

#endif
