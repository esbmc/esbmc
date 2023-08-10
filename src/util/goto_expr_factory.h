#pragma once

#include "goto-symex/symex_target_equation.h"

/**
 * @brief Create a constant_int expression
 *
 * @param value: numeric value of the constant
 * @param type: type of the expression
 * @return int negative means error, positive is the quantity to unroll
 */
expr2tc create_value_expr(int value, type2tc type);

/**
 * @brief Create a lessthanequal expression of the form LHS <= RHS
 * The comparation will use LHS for typecast
 *
 * @param lhs: left operator
 * @param rhs: right operator
 * @return int negative means error, positive is the quantity to unroll
 */
expr2tc create_lessthanequal_relation(expr2tc &lhs, expr2tc &rhs);

/**
 * @brief Create a lessthanequal expression of the form LHS <= RHS
 * The comparation will use LHS for typecast
 *
 * @param lhs: left operator
 * @param rhs: right operator
 * @return int negative means error, positive is the quantity to unroll
 */
expr2tc create_greaterthanequal_relation(expr2tc &lhs, expr2tc &rhs);
