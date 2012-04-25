#include "std_expr.h"
#include "std_types.h"
#include "irep2.h"

bool real_migrate_type(const typet &type, type2tc &new_type);
bool migrate_type(const typet &type, type2tc &new_type);
bool migrate_expr(const exprt &expr, expr2tc &new_expr);
