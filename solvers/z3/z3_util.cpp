/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>

#include "z3_conv.h"

//#define DEBUG

/*******************************************************************
 Function: z3_convt::print_data_types

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void z3_convt::print_data_types(Z3_ast operand0, Z3_ast operand1)
{
  Z3_type_ast a, b;

  a = Z3_get_type(z3_ctx, operand0);
  std::cout << "operand0 type:" << std::endl;
  std::cout << Z3_get_symbol_string(z3_ctx,Z3_get_type_name(z3_ctx, a)) << std::endl;

  b = Z3_get_type(z3_ctx, operand1);
  std::cout << "operand1:" << std::endl;
  std::cout << Z3_get_symbol_string(z3_ctx,Z3_get_type_name(z3_ctx, b)) << std::endl;
}

/*******************************************************************
 Function: z3_convt::show_bv_size

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void z3_convt::show_bv_size(Z3_ast operand)
{
  Z3_type_ast a;

  a = Z3_get_type(z3_ctx, operand);
  std::cout << "operand type: ";
  std::cout << Z3_get_symbol_string(z3_ctx,Z3_get_type_name(z3_ctx, a)) << std::endl;
  std::cout << "operand size: ";
  std::cout << Z3_get_bv_type_size(z3_ctx, a) << std::endl;
}

/*******************************************************************
 Function: z3_convt::check_all_types

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool z3_convt::check_all_types(const typet &type)
{
  if (type.id()=="bool" || type.id()=="signedbv" || type.id()=="unsignedbv" ||
	  type.id()=="symbol" || type.id()=="empty" || type.id() == "fixedbv" ||
	  type.id()=="array" || type.id()=="struct" || type.id()=="pointer" ||
	  type.id()=="union" || type.id()=="code")
  {
    return true;
  }

  return false;
}

/*******************************************************************
 Function: z3_convt::is_bv

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool z3_convt::is_bv(const typet &type)
{
  if (type.id()=="signedbv" || type.id()=="unsignedbv" ||
	  type.id() == "fixedbv")
    return true;

  return false;
}

/*******************************************************************
 Function: z3_convt::is_signed

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool z3_convt::is_signed(const typet &type)
{
  if (type.id()=="signedbv" || type.id()=="fixedbv")
    return true;

  return false;
}

/*******************************************************************
 Function: z3_convt::convert_number

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

Z3_ast z3_convt::convert_number(int64_t value, u_int width, bool type)
{

  if (int_encoding)
    return convert_number_int(value, width, type);
  else
    return convert_number_bv(value, width, type);
}

Z3_ast z3_convt::convert_number_int(int64_t value, u_int width, bool type)
{

  if (type)
    return z3_api.mk_int(z3_ctx, value);
  else
    return z3_api.mk_unsigned_int(z3_ctx, value);
}

Z3_ast z3_convt::convert_number_bv(int64_t value, u_int width, bool type)
{

  if (type)
    return Z3_mk_int(z3_ctx, value, Z3_mk_bv_type(z3_ctx, width));
  else
    return Z3_mk_unsigned_int(z3_ctx, value, Z3_mk_bv_type(z3_ctx, width));
}

/*******************************************************************
 Function: z3_convt::itos

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string z3_convt::itos(int i)
{
  std::stringstream s;
  s << i;

  return s.str();
}

void
z3_convt::debug_label_formula(std::string name, Z3_ast formula)
{
  unsigned &num = debug_label_map[name];
  std::string the_name = "__ESBMC_" + name + itos(num);
  num++;

  Z3_sort sort = Z3_get_sort(z3_ctx, formula);
  Z3_ast sym = z3_api.mk_var(z3_ctx, the_name.c_str(), sort);
  Z3_ast eq = Z3_mk_eq(z3_ctx, sym, formula);
  assert_formula(eq);
  return;
}
