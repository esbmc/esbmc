#include "std_expr.h"
#include "std_types.h"
#include "irep2.h"

void real_migrate_type(const typet &type, type2tc &new_type);
void migrate_type(const typet &type, type2tc &new_type);
void migrate_expr(const exprt &expr, expr2tc &new_expr);
typet migrate_type_back(const type2tc &ref);
exprt migrate_expr_back(const expr2tc &ref);
