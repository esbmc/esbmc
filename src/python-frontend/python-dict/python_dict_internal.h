#pragma once

// Shared include bundle for the python-dict module translation units.  This is
// the exact set of headers the original monolithic python_dict_handler.cpp
// pulled in, so every split unit sees the same declarations it did before the
// refactor.
#include <python-frontend/json_utils.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/context.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_code.h>

#include <algorithm>
#include <functional>
#include <sstream>
