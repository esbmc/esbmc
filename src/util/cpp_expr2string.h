#ifndef CPROVER_CPP_EXPR2STRING_H
#define CPROVER_CPP_EXPR2STRING_H

#include <string>

class exprt;
class namespacet;
class typet;

std::string cpp_expr2string(const exprt &expr, const namespacet &ns);
std::string cpp_type2string(const typet &type, const namespacet &ns);

#endif
