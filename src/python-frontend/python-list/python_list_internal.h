#pragma once

// Shared include bundle for the python-list module translation units.  This is
// the exact set of headers the original monolithic python_list.cpp pulled in,
// so every split unit sees the same declarations it did before the refactor.
#include <python-frontend/python_list.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/tuple_handler.h>
#include <util/c_types.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/type_utils.h>
#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <python-frontend/python_frontend_limits.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/mp_arith.h>
#include <util/python_types.h>
#include <util/symbolic_types.h>
#include <util/config.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/type_byte_size.h>
#include <string>
#include <functional>

#include "list_type_inference.h"
