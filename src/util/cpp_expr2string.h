
#pragma once

#include <util/c_expr2string.h>

std::string
cpp_expr2string(const exprt &expr, const namespacet &ns, unsigned flags = 0);

std::string
cpp_type2string(const typet &type, const namespacet &ns, unsigned flags = 0);
