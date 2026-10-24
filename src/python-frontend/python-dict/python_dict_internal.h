#pragma once

// Shared include bundle for the python-dict module translation units.  This is
// the exact set of headers the original monolithic python_dict_handler.cpp
// pulled in, so every split unit sees the same declarations it did before the
// refactor.
#include <python-frontend/json_utils.h>
#include <python-frontend/python-dict/python_dict_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/tuple/tuple_handler.h>
#include <python-frontend/type/type_handler.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/irep/migrate.h>
#include <util/lang/python_types.h>
#include <util/irep/std_code.h>

#include <algorithm>
#include <functional>
#include <sstream>

/// The source literal of a constant dict key, e.g. `1` or `"a"`, or empty for
/// any other key. A value's recorded type is filed under it, so a read at a
/// constant key consults only the values written there.
inline std::string dict_key_id(const nlohmann::json &key)
{
  return key.is_object() && key.value("_type", "") == "Constant"
           ? key["value"].dump()
           : std::string();
}
