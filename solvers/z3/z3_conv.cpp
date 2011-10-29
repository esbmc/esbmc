/*******************************************************************
   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <assert.h>
#include <ctype.h>
#include <fstream>
#include <sstream>
#include <std_expr.h>
#include <arith_tools.h>
#include <std_types.h>
#include <config.h>
#include <i2string.h>
#include <expr_util.h>
#include <string2array.h>
#include <pointer_offset_size.h>
#include <find_symbols.h>
#include <prefix.h>
#include <solvers/flattening/boolbv_width.h>
#include <fixedbv.h>
#include <solvers/flattening/boolbv.h>
#include <solvers/flattening/boolbv_type.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

//static Z3_ast core[Z3_UNSAT_CORE_LIMIT];
static std::vector<Z3_ast> core_vector;
static u_int unsat_core_size = 0;
static u_int number_of_assumptions = 0, assumptions_status = 0;

extern void finalize_symbols(void);

//#define DEBUG

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                        "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

/*******************************************************************
   Function: z3_convt::get_z3_core_size

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

z3_convt::~z3_convt()
{
  if (z3_prop.smtlib) {
    std::ofstream temp_out;
    Z3_string smt_lib_str, logic;
    Z3_ast assumpt_array[z3_prop.assumpt.size() + 1], formula;
    formula = Z3_mk_true(z3_ctx);

    for (unsigned i = 0; i < z3_prop.assumpt.size(); i++)
      assumpt_array[i] = z3_prop.assumpt.at(i);

    if (int_encoding)
      logic = "QF_AUFLIRA";
    else
      logic = "QF_AUFBV";

    //Z3_set_ast_print_mode(z3_ctx, Z3_PRINT_SMTLIB_COMPLIANT);
    smt_lib_str = Z3_benchmark_to_smtlib_string(z3_ctx, "ESBMC", logic,
                                    "unknown", "", z3_prop.assumpt.size(),
                                    assumpt_array, formula);

    temp_out.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);

    temp_out << smt_lib_str << std::endl;
  }

  if (!uw)
    Z3_pop(z3_ctx, 1);

  // Experimental: if we're handled say, 10,000 ileaves, refresh the z3 ctx.
  num_ctx_ileaves++;

  if (num_ctx_ileaves == 10000) {
    num_ctx_ileaves = 0;
    Z3_del_context(z3_ctx);
    finalize_symbols();
    Z3_reset_memory();
    z3_ctx = z3_api.mk_proof_context(!s_relevancy, s_is_uw);
  }
}

/*******************************************************************
   Function: z3_convt::get_z3_core_size

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

uint
z3_convt::get_z3_core_size(void)
{
  return unsat_core_size;
}

/*******************************************************************
   Function: z3_convt::get_z3_number_of_assumptions

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

uint
z3_convt::get_z3_number_of_assumptions(void)
{
  return assumptions_status;
}

/*******************************************************************
   Function: z3_convt::set_z3_core_size

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_z3_core_size(uint val)
{
  if (val)
    max_core_size = val;
}

/*******************************************************************
   Function: z3_convt::set_z3_encoding

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_z3_encoding(bool enc)
{
  int_encoding = enc;
}

/*******************************************************************
   Function: z3_convt::set_smtlib

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_smtlib(bool smt)
{
  z3_prop.smtlib = smt;
}

/*******************************************************************
   Function: z3_convt::get_z3_encoding

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::get_z3_encoding(void) const
{
  return int_encoding;
}

/*******************************************************************
   Function: z3_convt::set_filename

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_filename(std::string file)
{
  filename = file;
}

/*******************************************************************
   Function: z3_convt::set_z3_ecp

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_z3_ecp(bool ecp)
{
  equivalence_checking = ecp;
}

/*******************************************************************
   Function: z3_convt::extract_magnitude

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

std::string
z3_convt::extract_magnitude(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(0, width / 2), true), 10);
}

/*******************************************************************
   Function: z3_convt::extract_fraction

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

std::string
z3_convt::extract_fraction(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(width / 2, width), false), 10);
}

/*******************************************************************
   Function: z3_convt::fixed_point

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

std::string
z3_convt::fixed_point(std::string v, unsigned width)
{
  DEBUGLOC;

  const int precision = 10000;
  std::string i, f, b, result;
  double integer, fraction, base;
  int i_int, f_int;

  i = extract_magnitude(v, width);
  f = extract_fraction(v, width);
  b = integer2string(power(2, width / 2), 10);

  integer = atof(i.c_str());
  fraction = atof(f.c_str());
  base = (atof(b.c_str()));

  fraction = (fraction / base);

  if (fraction < 0)
    fraction = -fraction;

  fraction = fraction * precision;

  i_int = (int)integer;
  f_int = (int)fraction + 1;

  if (fraction != 0)
    result = itos(i_int * precision + f_int) + "/" + itos(precision);
  else
    result = itos(i_int);

  return result;
}


/*******************************************************************
   Function: z3_convt::generate_assumptions

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::generate_assumptions(const exprt &expr, const Z3_ast &result)
{
  DEBUGLOC

  std::string literal;
  static bool is_first_literal = true;

  literal = expr.op0().get_string("identifier").c_str();
  int pos = 0;

  for (u_int i = 0; i < literal.size(); i++)
  {
    if (literal.at(i) != '&')
      ++pos;
    else
      break;
  }

  literal.erase(pos, literal.size() - pos);

  if (unsat_core_size) {
    unsigned int i;

    for (i = 0; i < core_vector.size(); i++)
    {
      std::string id = Z3_ast_to_string(z3_ctx, core_vector.at(i));

      if (id.find(literal.c_str()) != std::string::npos)
	return;
    }
    assumptions.push_back(Z3_mk_not(z3_ctx, result));
  } else if (is_first_literal)   {
    is_first_literal = false;
    return;
  } else
    assumptions.push_back(Z3_mk_not(z3_ctx, result));

  ++number_of_assumptions;
}

/*******************************************************************
   Function: z3_convt::store_sat_assignments

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::store_sat_assignments(Z3_model m)
{
  unsigned num_constants, i;

  num_constants = Z3_get_model_num_constants(z3_ctx, m);

  for (i = 0; i < num_constants; i++)
  {
    std::string variable;
    Z3_symbol name;
    Z3_ast app, val;

    Z3_func_decl cnst = Z3_get_model_constant(z3_ctx, m, i);
    name = Z3_get_decl_name(z3_ctx, cnst);
    variable = Z3_get_symbol_string(z3_ctx, name);
    app = Z3_mk_app(z3_ctx, cnst, 0, 0);
    val = app;
    Z3_eval(z3_ctx, m, app, &val);
    map_vars.insert(std::pair<std::string, Z3_ast>(variable, val));
  }

  z3_prop.map_prop_vars = map_vars;
}

/*******************************************************************
   Function: z3_convt::check2_z3_properties

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_lbool
z3_convt::check2_z3_properties(void)
{
  DEBUGLOC

  assert(number_of_assumptions == assumptions.size());

  Z3_model m = 0;
  Z3_lbool result;
  unsigned i;
  Z3_ast proof, core[number_of_assumptions],
         assumptions_core[number_of_assumptions];
  std::string literal;

  assumptions_status = number_of_assumptions;

  if (uw)
    for (i = 0; i < assumptions.size(); i++)
      assumptions_core[i] = assumptions.at(i);

  try
  {
    if (uw)
      result = Z3_check_assumptions(z3_ctx, number_of_assumptions,
                             assumptions_core, &m, &proof, &unsat_core_size,
                             core);
    else
      result = Z3_check_and_get_model(z3_ctx, &m);
  }
  catch (std::string &error_str)
  {
    error(error_str);
    return Z3_L_UNDEF;
  }

  catch (const char *error_str)
  {
    error(error_str);
    return Z3_L_UNDEF;
  }

  catch (std::bad_alloc)
  {
    error("Out of memory");
    return Z3_L_UNDEF;
  }


  if (result == Z3_L_TRUE) {
    store_sat_assignments(m);
    //printf("Counterexample:\n");
    //z3_api.display_model(z3_ctx, stdout, m);
  } else if (uw && result == Z3_L_FALSE)   {
    for (i = 0; i < unsat_core_size; ++i)
    {
      std::string id = Z3_ast_to_string(z3_ctx, core[i]);
      if (id.find("false") != std::string::npos) {
	result = z3_api.check2(z3_ctx, Z3_L_TRUE);
	if (result == Z3_L_TRUE) {
	  store_sat_assignments(m);
	}
	unsat_core_size = 0;
	return result;
      }
      core_vector.push_back(core[i]);
    }
  }

  number_of_assumptions = 0;

  return result;
}

/*******************************************************************
   Function: z3_convt::is_ptr

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::is_ptr(const typet &type)
{
  return type.id() == "pointer" || type.id() == "reference";
}

/*******************************************************************
   Function: z3_convt::select_pointer_value

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::select_pointer_value(Z3_ast object, Z3_ast offset, Z3_ast &bv)
{
  DEBUGLOC;

  bv = Z3_mk_select(z3_ctx, object, offset);
  return false;
}

/*******************************************************************
   Function: z3_convt::create_z3_array_var

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::create_array_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  Z3_type_ast tuple_type, array_of_array_type;
  unsigned width;

  if (type.subtype().id() == "bool") {
    if (int_encoding)
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx),
                            Z3_mk_bool_type(z3_ctx));
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                         config.ansi_c.int_width),
                         Z3_mk_bool_type(z3_ctx));
  } else if (type.subtype().id() == "fixedbv")   {
    if (boolbv_get_width(type.subtype(), width))
      return true;

    if (int_encoding)
      bv  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx),
                             Z3_mk_real_type(z3_ctx));
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                         config.ansi_c.int_width),
                         Z3_mk_bv_type(z3_ctx, width));
  } else if (type.subtype().id() == "struct")   {
    if (create_struct_type(type.subtype(), tuple_type))
      return true;

    if (int_encoding)
      bv  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx), tuple_type);
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                            config.ansi_c.int_width),
                            tuple_type);
  } else if (type.subtype().id() == "union")   {
    if (create_union_type(type.subtype(), tuple_type))
      return true;

    if (int_encoding)
      bv  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx), tuple_type);
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                            config.ansi_c.int_width),
                            tuple_type);
  } else if (type.subtype().id() == "array")   { // array of array
    if (create_array_type(type.subtype(), array_of_array_type))
      return true;

    if (int_encoding)
      bv  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(
                               z3_ctx), array_of_array_type);
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                            config.ansi_c.int_width),
                            array_of_array_type);
  } else   {
    if (type.subtype().id() == "pointer") {
      if (boolbv_get_width(type.subtype().subtype(), width)) {
	if (type.subtype().subtype().id() == "empty"
	    || type.subtype().subtype().id() == "symbol")
	  width = config.ansi_c.int_width;
	else
	  return true;
      }
    } else   {
      if (boolbv_get_width(type.subtype(), width))
	return true;
    }

    if (int_encoding)
      bv  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx),
                             Z3_mk_int_type(z3_ctx));
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                            config.ansi_c.int_width),
                            Z3_mk_bv_type(z3_ctx, width));
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::create_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::create_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  unsigned width = config.ansi_c.int_width;

  if (type.id() == "bool") {
    bv = Z3_mk_bool_type(z3_ctx);
  } else if (type.id() == "signedbv" || type.id() == "unsignedbv" ||
             type.id() == "c_enum" || type.id() == "incomplete_c_enum") {
    if (boolbv_get_width(type, width))
      return true;

    if (int_encoding)
      bv = Z3_mk_int_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, width);
  } else if (type.id() == "fixedbv")   {
    if (boolbv_get_width(type, width))
      return true;

    if (int_encoding)
      bv = Z3_mk_real_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, width);
  } else if (type.id() == "array")     {
    if (type.subtype().id() == "struct") {
      if (create_struct_type(type.subtype(), bv))
	return true;

      if (int_encoding) {
	bv = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx), bv);
	return false;
      } else   {
	bv =
	  Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
	                                         config.ansi_c.int_width), bv);
	return false;
      }
    } else if (type.subtype().id() == "array")     {
      if (create_type(type.subtype(), bv))
	return true;

      if (int_encoding) {
	bv = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx), bv);
	return false;
      } else   {
	bv =
	  Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
	                                         config.ansi_c.int_width), bv);
	return false;
      }
    } else if (type.subtype().id() == "pointer")     {
      if (boolbv_get_width(type.subtype().subtype(), width)) {
	if (type.subtype().subtype().id() == "empty" ||
	    type.subtype().subtype().id() == "symbol")
	  width = config.ansi_c.int_width;
	else
	  return true;
      }
      if (!width) width = config.ansi_c.int_width;
    } else if (type.subtype().id() == "bool")     {
      if (int_encoding)
	bv = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx),
	                      Z3_mk_bool_type(z3_ctx));
      else
	bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
	                      config.ansi_c.int_width),
	                      Z3_mk_bool_type(z3_ctx));

      return false;
    } else   {
      if (boolbv_get_width(type.subtype(), width))
	return true;
    }

    if (int_encoding)
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx),
                            Z3_mk_int_type(z3_ctx));
    else
      bv = Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                            config.ansi_c.int_width),
                            Z3_mk_bv_type(z3_ctx, width));
  } else if (type.id() == "struct")     {
    if (create_struct_type(type, bv))
      return true;
  } else if (type.id() == "union")     {
    if (create_union_type(type, bv))
      return true;
  } else if (type.id() == "pointer")     {
    if (create_pointer_type(type, bv)) {
      if (type.subtype().id() == "symbol") {
	if (int_encoding)
	  bv = Z3_mk_int_type(z3_ctx);
	else
	  bv = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);

	return false;
      } else
	return true;
    }
  } else if (type.id() == "symbol" || type.id() == "empty")     {
    if (int_encoding)
      bv = Z3_mk_int_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  } else
    return true;

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::create_struct_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::create_struct_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  Z3_symbol mk_tuple_name, *proj_names;
  std::string struct_name;
  Z3_type_ast *proj_types;
  Z3_const_decl_ast mk_tuple_decl, *proj_decls;
  u_int size_of_struct;

  const struct_typet &struct_type = to_struct_type(type);

  const struct_typet::componentst &components =
    struct_type.components();

  assert(components.size() > 0);
  size_of_struct = components.size();
  proj_names = new Z3_symbol[size_of_struct];
  proj_types = new Z3_type_ast[size_of_struct];
  proj_decls = new Z3_const_decl_ast[size_of_struct];

  struct_name = "struct_type_" + type.get_string("tag");
  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, struct_name.c_str());

  u_int i = 0;
  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (create_type(it->type(), proj_types[i]))
      return true;

    proj_names[i] = Z3_mk_string_symbol(z3_ctx, it->get("name").c_str());
  }

  bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, i, proj_names, proj_types,
                        &mk_tuple_decl, proj_decls);

  delete[] proj_names;
  delete[] proj_types;
  delete[] proj_decls;

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::create_union_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::create_union_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  Z3_symbol mk_tuple_name, *proj_names;
  std::string union_name;
  Z3_type_ast *proj_types;
  Z3_const_decl_ast mk_tuple_decl, *proj_decls;
  u_int size_of_union;

  const struct_typet &struct_type = to_struct_type(type);

  const struct_typet::componentst &components =
    struct_type.components();

  assert(components.size() > 0);
  size_of_union = components.size() + 1;
  proj_names = new Z3_symbol[size_of_union];
  proj_types = new Z3_type_ast[size_of_union];
  proj_decls = new Z3_const_decl_ast[size_of_union];

  union_name = "union_type_" + type.get_string("tag");
  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, union_name.c_str());

  u_int i = 0;
  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    proj_names[i] = Z3_mk_string_symbol(z3_ctx, it->get("name").c_str());
    if (create_type(it->type(), proj_types[i]))
      return true;
  }

  //The id field records the last value written to the union
  proj_names[size_of_union - 1] = Z3_mk_string_symbol(z3_ctx, "id");

  if (int_encoding)
    proj_types[size_of_union - 1] = Z3_mk_int_type(z3_ctx);
  else
    proj_types[size_of_union - 1] = Z3_mk_bv_type(z3_ctx,
                                                  config.ansi_c.int_width);


  bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, size_of_union, proj_names,
                        proj_types, &mk_tuple_decl, proj_decls);

  delete[] proj_names;
  delete[] proj_types;
  delete[] proj_decls;

  return false;
}

/*******************************************************************
   Function: z3_convt::create_enum_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::create_enum_type(Z3_type_ast &bv)
{
  DEBUGLOC;

  if (int_encoding)
    bv = Z3_mk_int_type(z3_ctx);
  else
    bv = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);

  return false;
}

/*******************************************************************
   Function: z3_convt::create_pointer_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/
bool
z3_convt::create_pointer_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  typet actual_type;
  Z3_symbol mk_tuple_name, proj_names[2];
  Z3_type_ast proj_types[2];
  Z3_const_decl_ast mk_tuple_decl, proj_decls[2];

  if (is_ptr(type.subtype())) {
    actual_type = select_pointer(type.subtype());
    if (create_type(select_pointer(type.subtype()), proj_types[0]))
      return true;
  } else if (type.subtype().id() == "code")     {
    actual_type = type.subtype();
    if (int_encoding)
      proj_types[0] = Z3_mk_int_type(z3_ctx);
    else
      proj_types[0] = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  } else if (check_all_types(type.subtype()))   {
    actual_type = type.subtype();
    if (create_type(type.subtype(), proj_types[0])) {
      if (type.subtype().id() == "symbol") {
	if (int_encoding)
	  proj_types[0] = Z3_mk_int_type(z3_ctx);
	else
	  proj_types[0] = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
      } else
	return true;
    }
  } else if (check_all_types(type))   {
    actual_type = type;
    if (create_type(type, proj_types[0])) {
      if (type.id() == "symbol") {
	if (int_encoding)
	  proj_types[0] = Z3_mk_int_type(z3_ctx);
	else
	  proj_types[0] = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
      } else
	return true;
    }
  } else   {
    return true;
  }

  DEBUGLOC;

  std::string name = "pointer_tuple_";
  std::stringstream s;

  if (int_encoding) {
    name += Z3_get_symbol_string(z3_ctx, Z3_get_type_name(z3_ctx, proj_types[0]));
  } else   {
    name += Z3_get_symbol_string(z3_ctx, Z3_get_type_name(z3_ctx, proj_types[0]));
    if (actual_type.id() != "bool") {
      if (actual_type.id() == "struct")
	s << "struct";
      else
	s << Z3_get_bv_type_size(z3_ctx, proj_types[0]);
    }
    name += s.str();
  }

  DEBUGLOC;

  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, name.c_str());
  proj_names[0] = Z3_mk_string_symbol(z3_ctx, "object");

  proj_names[1] = Z3_mk_string_symbol(z3_ctx, "index");

  if (int_encoding)
    proj_types[1] = Z3_mk_int_type(z3_ctx);
  else
    proj_types[1] = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);

  bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, 2, proj_names, proj_types,
                        &mk_tuple_decl, proj_decls);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_identifier

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_identifier(const std::string &identifier, const typet &type,
  Z3_ast &bv)
{
  DEBUGLOC;

  Z3_type_ast type_var;
  unsigned width;

  if (type.id() == "bool") {
    bv = z3_api.mk_bool_var(z3_ctx, identifier.c_str());
  } else if (type.id() == "signedbv")     {
    if (boolbv_get_width(type, width))
      return true;

    if (int_encoding)
      bv = z3_api.mk_int_var(z3_ctx, identifier.c_str());
    else
      bv =
        z3_api.mk_var(z3_ctx, identifier.c_str(), Z3_mk_bv_type(z3_ctx, width));
  } else if (type.id() == "unsignedbv")     {
    if (boolbv_get_width(type, width))
      return true;

    if (int_encoding) {
      Z3_ast formula;
      bv = z3_api.mk_int_var(z3_ctx, identifier.c_str());
      formula = Z3_mk_ge(z3_ctx, bv, z3_api.mk_int(z3_ctx, 0));
      Z3_assert_cnstr(z3_ctx, formula);
      if (z3_prop.smtlib)
	z3_prop.assumpt.push_back(formula);
    } else
      bv =
        z3_api.mk_var(z3_ctx, identifier.c_str(), Z3_mk_bv_type(z3_ctx, width));
  } else if (type.id() == "fixedbv")    {
    if (boolbv_get_width(type, width))
      return true;

    if (int_encoding)
      bv = z3_api.mk_real_var(z3_ctx, identifier.c_str());
    else
      bv =
        z3_api.mk_var(z3_ctx, identifier.c_str(), Z3_mk_bv_type(z3_ctx, width));
  } else if (type.id() == "array")     {
    if (create_array_type(type, type_var))
      return true;

    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), type_var);
  } else if (type.id() == "pointer" || type.id() == "code")       {
    if (create_pointer_type(type, type_var))
      return true;

    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), type_var);
  } else if (type.id() == "struct")     {
    if (create_struct_type(type, type_var))
      return true;

    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), type_var);
  } else if (type.id() == "union")     {
    if (create_union_type(type, type_var))
      return true;

    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), type_var);
  } else if (type.id() == "c_enum")     {
    if (create_enum_type(type_var))
      return true;

    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), type_var);
  } else   {
    return true;
  }

  //throw "convert_identifier: " + type.id_string() + " is unsupported";

  DEBUGLOC;

  return false;
}

/*******************************************************************\

   Function: z3_convt::is_in_cache Inputs:

   Outputs:

   Purpose:

\*******************************************************************/

bool
z3_convt::is_in_cache(const exprt &expr)
{
  bv_cachet::const_iterator cache_result = bv_cache.find(expr);
  if (cache_result != bv_cache.end()) {
    //std::cout << "Cache hit on " << expr.pretty() << std::endl;
    return true;
  }

  return false;
}

/*******************************************************************\

   Function: z3_convt::convert_bv Inputs:

   Outputs:

   Purpose:

\*******************************************************************/

bool
z3_convt::convert_bv(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

#if 1
  bv_cachet::const_iterator cache_result = bv_cache.find(expr);
  if (cache_result != bv_cache.end()) {
    bv = cache_result->second;
    return false;
  }
#endif
  if (convert_z3_expr(expr, bv))
    return true;

  // insert into cache
  bv_cache.insert(std::pair<const exprt, Z3_ast>(expr, bv));

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::read_cache

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::read_cache(const exprt &expr, Z3_ast &bv)
{
  std::string symbol;

  symbol = expr.get_string("identifier");

  for (z3_cachet::const_iterator it = z3_cache.begin();
       it != z3_cache.end(); it++)
  {
    if (symbol.compare((*it).second.c_str()) == 0) {
      if (convert_bv((*it).first, bv))
	return true;

      return false;
    }
  }

  if (convert_bv(expr, bv))
    return true;

  return false;
}

/*******************************************************************
   Function: z3_convt::write_cache

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::write_cache(const exprt &expr)
{
  std::string symbol, identifier;

  identifier = expr.get_string("identifier");

  for (std::string::const_iterator it = identifier.begin(); it
       != identifier.end(); it++)
  {
    char ch = *it;

    if (isalnum(ch) || ch == '$' || ch == '?') {
      symbol += ch;
    } else if (ch == '#')   {
      z3_cache.insert(std::pair<const exprt, std::string>(expr, symbol));
      return false;
    }
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_lt

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_lt(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];
  const exprt &op0 = expr.op0();
  const exprt &op1 = expr.op1();

  if (convert_bv(op0, operand[0]))
    return Z3_mk_false(z3_ctx);

  if (convert_bv(op1, operand[1]))
    return Z3_mk_false(z3_ctx);

  if (op0.type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (op1.type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (op0.id() == "typecast" && op0.type().id() == "signedbv") {
    const exprt &object = expr.op0().operands()[0];
    std::string value;
    unsigned width;

    if (op1.id() == "constant" && object.type().id() == "unsignedbv") {
      value = integer2string(binary2integer(expr.op1().get_string(
                                            "value"), false), 10);

      if (boolbv_get_width(expr.op1().type(), width))
	return Z3_mk_false(z3_ctx);

      if (int_encoding) {
	operand[1] = z3_api.mk_int(z3_ctx, atoi(value.c_str()));
	bv = Z3_mk_lt(z3_ctx, operand[0], operand[1]);
      } else   {
	operand[1] = convert_number(atoi(value.c_str()), width, false);
	bv = Z3_mk_bvult(z3_ctx, operand[0], operand[1]);
      }
      return bv;
    }
  }

  if (int_encoding) {
    bv = Z3_mk_lt(z3_ctx, operand[0], operand[1]);
  } else   {
    if (op1.type().id() == "signedbv" || op1.type().id() == "fixedbv"
        || op1.type().id() == "pointer")
      bv = Z3_mk_bvslt(z3_ctx, operand[0], operand[1]);
    else if (op1.type().id() == "unsignedbv")
      bv = Z3_mk_bvult(z3_ctx, operand[0], operand[1]);
  }

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_gt

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/
Z3_ast
z3_convt::convert_gt(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);
  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  //workaround: bug in the front-end
  //to check it, run example ex10.c of NEC
  if (expr.op0().id() == "typecast" && expr.op0().type().id() == "signedbv") {
    const exprt &object = expr.op0().operands()[0];
    std::string value;
    unsigned width;

    if (expr.op1().id() == "constant" && object.type().id() == "unsignedbv") {
      value = integer2string(binary2integer(expr.op1().get_string(
                                            "value"), false), 10);

      if (boolbv_get_width(expr.op1().type(), width))
	return Z3_mk_false(z3_ctx);

      if (int_encoding) {
	operand[1] = z3_api.mk_int(z3_ctx, atoi(value.c_str()));
	bv = Z3_mk_gt(z3_ctx, operand[0], operand[1]);
      } else   {
	operand[1] = convert_number(atoi(value.c_str()), width, false);
	bv = Z3_mk_bvugt(z3_ctx, operand[0], operand[1]);
      }
      return bv;
    }
  }

  if (int_encoding) {
    bv = Z3_mk_gt(z3_ctx, operand[0], operand[1]);
  } else   {
    if (expr.op1().type().id() == "signedbv" || expr.op1().type().id() ==
        "fixedbv"
        || expr.op1().type().id() == "pointer")
      bv = Z3_mk_bvsgt(z3_ctx, operand[0], operand[1]);
    else if (expr.op1().type().id() == "unsignedbv")
      bv = Z3_mk_bvugt(z3_ctx, operand[0], operand[1]);
  }

  return bv;
}


/*******************************************************************
   Function: z3_convt::convert_le

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_le(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);
  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (expr.op0().id() == "typecast" && expr.op0().type().id() == "signedbv") {
    const exprt &object = expr.op0().operands()[0];
    std::string value;
    unsigned width;

    if (expr.op1().id() == "constant" && object.type().id() == "unsignedbv") {
      value = integer2string(binary2integer(expr.op1().get_string(
                                            "value"), false), 10);

      if (boolbv_get_width(expr.op1().type(), width))
	return Z3_mk_false(z3_ctx);

      if (int_encoding) {
	operand[1] = z3_api.mk_int(z3_ctx, atoi(value.c_str()));
	bv = Z3_mk_le(z3_ctx, operand[0], operand[1]);
      } else   {
	operand[1] = convert_number(atoi(value.c_str()), width, false);
	bv = Z3_mk_bvule(z3_ctx, operand[0], operand[1]);
      }

      return bv;
    }
  }

  if (int_encoding) {
    bv = Z3_mk_le(z3_ctx, operand[0], operand[1]);
  } else   {
    if (expr.op1().type().id() == "signedbv" || expr.op1().type().id() ==
        "fixedbv"
        || expr.op1().type().id() == "pointer")
      bv = Z3_mk_bvsle(z3_ctx, operand[0], operand[1]);
    else if (expr.op1().type().id() == "unsignedbv")
      bv = Z3_mk_bvule(z3_ctx, operand[0], operand[1]);
  }

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_ge

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_ge(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);

  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);


  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (expr.op0().id() == "typecast" && expr.op0().type().id() == "signedbv") {
    const exprt &object = expr.op0().operands()[0];
    std::string value;
    unsigned width;

    if (expr.op1().id() == "constant" && object.type().id() == "unsignedbv") {
      value = integer2string(binary2integer(expr.op1().get_string(
                                            "value"), false), 10);

      if (boolbv_get_width(expr.op1().type(), width))
	return Z3_mk_false(z3_ctx);

      if (int_encoding) {
	operand[1] = z3_api.mk_int(z3_ctx, atoi(value.c_str()));
	bv = Z3_mk_ge(z3_ctx, operand[0], operand[1]);
      } else   {
	operand[1] = convert_number(atoi(value.c_str()), width, false);
	bv = Z3_mk_bvuge(z3_ctx, operand[0], operand[1]);
      }

      return bv;
    }
  }

  if (int_encoding) {
    bv = Z3_mk_ge(z3_ctx, operand[0], operand[1]);
  } else   {
    if (expr.op1().type().id() == "signedbv" || expr.op1().type().id() ==
        "fixedbv"
        || expr.op1().type().id() == "pointer")
      bv = Z3_mk_bvsge(z3_ctx, operand[0], operand[1]);
    else if (expr.op1().type().id() == "unsignedbv")
      bv = Z3_mk_bvuge(z3_ctx, operand[0], operand[1]);
  }

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_eq

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_eq(const exprt &expr)
{
  DEBUGLOC;

  //std::cout << "expr.pretty(): " << expr.pretty() << std::endl;

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];
  const exprt &op0 = expr.op0();
  const exprt &op1 = expr.op1();

  if (op0.type().id() == "array") {
    if (write_cache(op0))
      return Z3_mk_false(z3_ctx);
  }

  if (op0.type().subtype().id() == "code" &&
		  op1.type().subtype().id() == "code") {
    if (!is_in_cache(op0) && !is_in_cache(op1))
      return Z3_mk_true(z3_ctx);
  }

  if (convert_bv(op0, operand[0]))
    return Z3_mk_false(z3_ctx);

  if (convert_bv(op1, operand[1]))
    return Z3_mk_false(z3_ctx);

  if (op0.type().id() == "pointer" && op1.type().id() == "pointer") {
    Z3_ast pointer[2], formula[2];

    if (op1.id() == "address_of")
      z3_cache.insert(std::pair<const exprt, std::string>(op1.op0(),
                                                          op0.get_string(
                                                            "identifier")));
    else
      z3_cache.insert(std::pair<const exprt, std::string>(op1,
                                                          op0.get_string(
                                                            "identifier")));


    pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 0);
    pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 0);

    if (expr.id() == "=")
      formula[0] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);
    else
      formula[0] = Z3_mk_distinct(z3_ctx, 2, pointer);

    pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);
    pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

    if (expr.id() == "=")
      formula[1] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);
    else
      formula[1] = Z3_mk_distinct(z3_ctx, 2, pointer);

    bv = Z3_mk_and(z3_ctx, 2, formula);

    //return bv;
  } else   {
    if (expr.id() == "=")
      bv = Z3_mk_eq(z3_ctx, operand[0], operand[1]);
    else
      bv = Z3_mk_distinct(z3_ctx, 2, operand);
  }

  DEBUGLOC;

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_invalid

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_invalid(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast bv, pointer, operand[2];

  if (expr.op0().id() == "address_of" ||
      (expr.op0().type().id() == "pointer" &&
       (expr.op0().type().subtype().id() == "symbol"
       || expr.op0().type().subtype().id() == "code")))
    return Z3_mk_false(z3_ctx);

  if (!is_in_cache(expr.op0()) && expr.op0().id() == "typecast")
    return Z3_mk_true(z3_ctx);

  if (!is_in_cache(expr.op0())) {
    if (expr.op0().id() == "+" && expr.op0().type().subtype().id() != "empty")
      return Z3_mk_true(z3_ctx);
    else
      return Z3_mk_false(z3_ctx);
  }

  //the subtype field of index is empty and generates seg. fault
  if (!is_in_cache(expr.op0()) && expr.op0().id() == "index")
    return Z3_mk_false(z3_ctx);

  if (convert_bv(expr.op0(), pointer)) //return pointer tuple
    return Z3_mk_false(z3_ctx);

  //@TODO
  if (expr.op0().id() == "member" && expr.op0().type().id() == "pointer" &&
      expr.op0().type().subtype().id() == "signedbv")
    return Z3_mk_false(z3_ctx);

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, pointer, 1);       //pointer
	                                                           // index

  operand[1] = convert_number(0, config.ansi_c.int_width, true);

  if (expr.op0().type().id() == "pointer" &&
      (expr.op0().type().subtype().id() == "signedbv"
       || expr.op0().type().subtype().id()
       == "empty")) {
    operand[1] = convert_number(-1, config.ansi_c.int_width, true);
    return Z3_mk_not(z3_ctx, Z3_mk_distinct(z3_ctx, 2, operand));
  }

  if (int_encoding)
    bv = Z3_mk_ge(z3_ctx, operand[0], operand[1]);
  else
    bv = Z3_mk_bvsge(z3_ctx, operand[0], operand[1]);

  DEBUGLOC;

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_same_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_same_object(const exprt &expr)
{
  DEBUGLOC;

  const exprt::operandst &operands = expr.operands();
  Z3_ast bv, operand[2], pointer[2], offset[2], formula[2], zero;
  const exprt &op0 = expr.op0();
  const exprt &op1 = expr.op1();

  if ((is_in_cache(op0) && !is_in_cache(op1))
      || (!is_in_cache(op0) && !is_in_cache(op1))) {
    //object is not in the cache and generates spurious counter-example
    return Z3_mk_false(z3_ctx);
  } else if (op0.id() == "address_of" && op1.id() == "constant")       {
    return Z3_mk_false(z3_ctx);     //TODO
  } else if (op0.id() == "symbol" && op1.id() == "address_of")       {
    const exprt &object = op1.operands()[0];
    const exprt &index = object.operands()[0];

    if (convert_bv(op0, pointer[0]))
      return Z3_mk_false(z3_ctx);

    if (convert_bv(op1, pointer[1]))
      return Z3_mk_false(z3_ctx);

    if (op1.type().id() == "pointer" && op1.type().subtype().id() == "union")
      return Z3_mk_false(z3_ctx);

    operand[0] = z3_api.mk_tuple_select(z3_ctx, pointer[0], 0);
    operand[1] = z3_api.mk_tuple_select(z3_ctx, pointer[1], 0);
    formula[0] = Z3_mk_eq(z3_ctx, operand[0], operand[1]);

    operand[0] = z3_api.mk_tuple_select(z3_ctx, pointer[0], 1);
    operand[1] = z3_api.mk_tuple_select(z3_ctx, pointer[1], 1);
    formula[1] = Z3_mk_eq(z3_ctx, operand[0], operand[1]);

    if (object.operands().size() > 0) {
      if (expr.op0().type().subtype().id() == "signedbv"
          && expr.op1().type().subtype().id() == "signedbv"
          && index.id() != "string-constant")
	formula[1] = formula[0];
    }

    bv = Z3_mk_and(z3_ctx, 2, formula);
  } else if (operands.size() == 2 &&
             is_ptr(operands[0].type()) &&
             is_ptr(operands[1].type())) {
    if (convert_bv(op0, pointer[0]))
      return Z3_mk_false(z3_ctx);
    if (convert_bv(op1, pointer[1]))
      return Z3_mk_false(z3_ctx);

    operand[0] = z3_api.mk_tuple_select(z3_ctx, pointer[0], 0);
    operand[1] = z3_api.mk_tuple_select(z3_ctx, pointer[1], 0);

    unsigned width_op0;

    if (boolbv_get_width(expr.op0().type().subtype(), width_op0))
      width_op0 = config.ansi_c.int_width;

    if (!int_encoding)
      if (op0.type().subtype().id() != "pointer" &&
          op1.type().subtype().id() == "empty")
	operand[1] = Z3_mk_extract(z3_ctx, width_op0 - 1, 0, operand[1]);

    formula[0] = Z3_mk_eq(z3_ctx, operand[0], operand[1]);

    zero = convert_number(0, config.ansi_c.int_width, true);

    operand[0] = z3_api.mk_tuple_select(z3_ctx, pointer[0], 1);
    operand[1] = z3_api.mk_tuple_select(z3_ctx, pointer[1], 1);

    if (int_encoding) {
      offset[0] = Z3_mk_ge(z3_ctx, operand[0], zero);
      offset[1] = Z3_mk_ge(z3_ctx, operand[1], zero);
    } else   {
      offset[0] = Z3_mk_bvsge(z3_ctx, operand[0], zero);
      offset[1] = Z3_mk_bvsge(z3_ctx, operand[1], zero);
    }

    formula[1] = Z3_mk_and(z3_ctx, 2, offset);
    bv = Z3_mk_and(z3_ctx, 2, formula);
  }

  DEBUGLOC;

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_dynamic_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_dynamic_object(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast bv, operand0, operand1;
  std::vector<unsigned> dynamic_objects;
  pointer_logic.get_dynamic_objects(dynamic_objects);

  if (dynamic_objects.empty())
    return Z3_mk_true(z3_ctx);
  else {
    unsigned width;

    if (convert_bv(expr.op0(), operand0))
      return Z3_mk_false(z3_ctx);

    if (expr.op0().type().id() == "pointer") {
      if (expr.op0().type().subtype().id() != "pointer") {
	if (boolbv_get_width(expr.op0().type().subtype(), width))
	  return Z3_mk_false(z3_ctx);
      } else if (expr.op0().type().subtype().subtype().id() != "pointer")     {
	if (boolbv_get_width(expr.op0().type().subtype().subtype(), width))
	  return Z3_mk_false(z3_ctx);
      } else
	width = config.ansi_c.int_width;
    } else   {
      if (boolbv_get_width(expr.op0().type(), width))
	return Z3_mk_false(z3_ctx);
    }

    if (expr.op0().type().id() == "pointer")
      operand0 = z3_api.mk_tuple_select(z3_ctx, operand0, 0);

    if (dynamic_objects.size() == 1) {
      Z3_ast tmp;
      operand1 = convert_number(dynamic_objects.front(), width, true);
      tmp = Z3_mk_eq(z3_ctx, operand0, operand1);
      Z3_assert_cnstr(z3_ctx, tmp);
      bv = Z3_mk_not(z3_ctx, tmp);
    } else   {
      unsigned i = 0, size;
      size = dynamic_objects.size() + 1;
      Z3_ast args[size];

      for (std::vector<unsigned>::const_iterator
           it = dynamic_objects.begin();
           it != dynamic_objects.end();
           it++, i++)
      {
	args[i] = Z3_mk_eq(z3_ctx, operand0, convert_number(*it, width, true));
	bv = Z3_mk_or(z3_ctx, i, args);
      }
    }
  }

  DEBUGLOC;

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_sum

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_sum(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, result[2], operand[2];
  unsigned width_op0, width_op1;

  if (expr.op0().type().id() == "array") {
    if (write_cache(expr.op0()))
      return Z3_mk_false(z3_ctx);
  }

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);
  ;

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);
  ;

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (boolbv_get_width(expr.op0().type(), width_op0))
    return Z3_mk_false(z3_ctx);

  if (boolbv_get_width(expr.op1().type(), width_op1))
    return Z3_mk_false(z3_ctx);

  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width_op0, operand[0]);
    operand[1] = Z3_mk_int2bv(z3_ctx, width_op1, operand[1]);
  }

  if (expr.op0().type().id() == "signedbv" && expr.op1().type().id() ==
      "signedbv")
    result[0] = Z3_mk_bvadd_no_overflow(z3_ctx, operand[0], operand[1], 1);
  else if (expr.op0().type().id() == "unsignedbv" && expr.op1().type().id() ==
           "unsignedbv")
    result[0] = Z3_mk_bvadd_no_overflow(z3_ctx, operand[0], operand[1], 0);

  result[1] = Z3_mk_bvadd_no_underflow(z3_ctx, operand[0], operand[1]);
  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, result));

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_sub

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_sub(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, result[2], operand[2];
  unsigned width_op0, width_op1;

  if (expr.op0().type().id() == "array") {
    if (write_cache(expr.op0()))
      return Z3_mk_false(z3_ctx);
  }
  //new change
  if (expr.op0().id() == "symbol" && expr.op1().id() == "address_of")
    return Z3_mk_false(z3_ctx);

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (boolbv_get_width(expr.op0().type(), width_op0))
    return Z3_mk_false(z3_ctx);

  if (boolbv_get_width(expr.op1().type(), width_op1))
    return Z3_mk_false(z3_ctx);

  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width_op0, operand[0]);
    operand[1] = Z3_mk_int2bv(z3_ctx, width_op1, operand[1]);
  }

  if (expr.op0().type().id() == "signedbv" && expr.op1().type().id() ==
      "signedbv")
    result[1] = Z3_mk_bvsub_no_underflow(z3_ctx, operand[0], operand[1], 1);
  else if (expr.op0().type().id() == "unsignedbv" && expr.op1().type().id() ==
           "unsignedbv")
    result[1] = Z3_mk_bvsub_no_underflow(z3_ctx, operand[0], operand[1], 0);

  result[0] = Z3_mk_bvsub_no_overflow(z3_ctx, operand[0], operand[1]);

  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, result));

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_mul

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_mul(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, result[2], operand[2];
  unsigned width_op0, width_op1;

  if (expr.op0().type().id() == "array") {
    if (write_cache(expr.op0()))
      return Z3_mk_false(z3_ctx);
    ;
  }

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);
  ;

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (convert_bv(expr.op1(), operand[1]))
    return Z3_mk_false(z3_ctx);
  ;

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (boolbv_get_width(expr.op0().type(), width_op0))
    return Z3_mk_false(z3_ctx);

  if (boolbv_get_width(expr.op1().type(), width_op1))
    return Z3_mk_false(z3_ctx);

  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width_op0, operand[0]);
    operand[1] = Z3_mk_int2bv(z3_ctx, width_op1, operand[1]);
  }

  if (expr.op0().type().id() == "signedbv" && expr.op1().type().id() ==
      "signedbv")
    result[1] = Z3_mk_bvmul_no_overflow(z3_ctx, operand[0], operand[1], 1);
  else if (expr.op0().type().id() == "unsignedbv" && expr.op1().type().id() ==
           "unsignedbv")
    result[1] = Z3_mk_bvmul_no_overflow(z3_ctx, operand[0], operand[1], 0);

  result[0] = Z3_mk_bvmul_no_underflow(z3_ctx, operand[0], operand[1]);

  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, result));

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_unary

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_unary(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast bv, operand;
  unsigned width;

  if (convert_bv(expr.op0(), operand))
    return Z3_mk_false(z3_ctx);
  ;

  if (expr.op0().type().id() == "pointer")
    operand = z3_api.mk_tuple_select(z3_ctx, operand, 1);

  if (boolbv_get_width(expr.op0().type(), width))
    return Z3_mk_false(z3_ctx);

  if (int_encoding)
    operand = Z3_mk_int2bv(z3_ctx, width, operand);

  bv = Z3_mk_not(z3_ctx, Z3_mk_bvneg_no_overflow(z3_ctx, operand));

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_typecast

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_typecast(const exprt &expr)
{
  DEBUGLOC;

  unsigned bits = atoi(expr.id().c_str() + 18);

  const exprt::operandst &operands = expr.operands();

  if (operands.size() != 1)
    throw "operand " + expr.id_string() + " takes one operand";

  Z3_ast bv, operand[3], mid, overflow[2], tmp, minus_one, two;
  u_int i, result = 1, width;
  std::string value;
  bool encoding = false;

  if (boolbv_get_width(expr.op0().type(), width))
    return Z3_mk_false(z3_ctx);

  if (bits >= width || bits == 0)
    throw "overflow-typecast got wrong number of bits";

  assert(bits <= 32);

  for (i = 0; i < bits; i++)
  {
    if (i == 31)
      result = (result - 1) * 2 + 1;
    else if (i < 31)
      result *= 2;
  }

  if (is_signed(expr.op0().type()))
    value = integer2string(binary2integer(expr.op0().get_string(
                                            "value"), true), 10);
  else
    value = integer2string(binary2integer(expr.op0().get_string(
                                            "value"), false), 10);

  if (convert_bv(expr.op0(), operand[0]))
    return Z3_mk_false(z3_ctx);

  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width, operand[0]);
    encoding = true;
    int_encoding = false;
  }

  if (expr.op0().type().id() == "fixedbv") {
    unsigned size = (width / 2) + 1;
    operand[0] = Z3_mk_extract(z3_ctx, width - 1, size - 1, operand[0]);
  }

  if (expr.op0().type().id() == "signedbv" || expr.op0().type().id() ==
      "fixedbv") {
    if (expr.op0().type().id() == "signedbv") {
      tmp = convert_number(result, width, true);
      two = convert_number(2, width, true);
      minus_one = convert_number(-1, width, true);
    } else   {
      tmp = convert_number(result, width / 2, true);
      two = convert_number(2, width / 2, true);
      minus_one = convert_number(-1, width / 2, true);
    }

    mid = Z3_mk_bvsdiv(z3_ctx, tmp, two);
    operand[1] = Z3_mk_bvsub(z3_ctx, mid, minus_one);
    operand[2] = Z3_mk_bvmul(z3_ctx, operand[1], minus_one);

    overflow[0] = Z3_mk_bvslt(z3_ctx, operand[0], operand[1]);
    overflow[1] = Z3_mk_bvsgt(z3_ctx, operand[0], operand[2]);
    bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, overflow));
  } else if (expr.op0().type().id() == "unsignedbv")     {
    operand[2] = convert_number(0, width, false);
    operand[1] = convert_number(result, width, false);
    overflow[0] = Z3_mk_bvult(z3_ctx, operand[0], operand[1]);
    overflow[1] = Z3_mk_bvuge(z3_ctx, operand[0], operand[2]);
    bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, overflow));
  }

  if (encoding) {
    int_encoding = true;
    encoding = false;
  }

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_rest_member

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_rest_member(const exprt &expr)
{
  DEBUGLOC;

  Z3_ast bv;

  if (convert_bv(expr, bv))
    return Z3_mk_false(z3_ctx);

  if (expr.get_string("component_name") == "is_zero") {
    bv = Z3_mk_not(z3_ctx, Z3_mk_eq(z3_ctx, Z3_mk_false(z3_ctx), bv));
    //std::cout << std::endl << __FUNCTION__ << "[" << __LINE__ << "]" <<
    //std::endl;
  }

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_rest_index

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_rest_index(const exprt &expr)
{
  DEBUGLOC;

  Z3_ast bv, operand0, operand1;

  if (expr.operands().size() != 2)
    throw "index takes two operands";

  const exprt &array = expr.op0();
  const exprt &index = expr.op1();

  if (convert_bv(array, operand0))
    return Z3_mk_false(z3_ctx);

  if (convert_bv(index, operand1))
    return Z3_mk_false(z3_ctx);

  DEBUGLOC;

  if (expr.op1().operands()[0].operands().size() == 0) {
    bv =
      Z3_mk_select(z3_ctx, operand0,
                   convert_number(0, config.ansi_c.int_width, true));
  } else if ((expr.op0().get_string("identifier").find("__ESBMC_alloc") !=
              std::string::npos
              && expr.op1().operands()[0].op0().id() != "index")) {
    bv =
      Z3_mk_select(z3_ctx, operand0,
                   convert_number(0, config.ansi_c.int_width, true));
  } else   {
    bv = Z3_mk_select(z3_ctx, operand0, operand1);
    bv = Z3_mk_eq(z3_ctx, bv, Z3_mk_false(z3_ctx));
  }


  DEBUGLOC;

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_memory_leak

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_memory_leak(const exprt &expr)
{
  DEBUGLOC;

  Z3_ast bv, operand0, operand1;

  if (expr.operands().size() != 2)
    throw "index takes two operands";

  const exprt &array = expr.op0();
  const exprt &index = expr.op1();

  if (convert_bv(array, operand0))
    return Z3_mk_false(z3_ctx);

  if (convert_bv(index, operand1))
    return Z3_mk_false(z3_ctx);

  bv = Z3_mk_select(z3_ctx, operand0, operand1);

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_rest

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

literalt
z3_convt::convert_rest(const exprt &expr)
{
  DEBUGLOC;

  //std::cout << "expr.pretty(): " << expr.pretty() << std::endl;

  literalt l = z3_prop.new_variable();
  Z3_ast formula, constraint;

  if (!assign_z3_expr(expr) && !ignoring_expr)
    return l;

  if (expr.id() == "<")
    constraint = convert_lt(expr);
  else if (expr.id() == ">")
    constraint = convert_gt(expr);
  else if (expr.id() == "<=")
    constraint = convert_le(expr);
  else if (expr.id() == ">=")
    constraint = convert_ge(expr);
  else if (expr.id() == "=" || expr.id() == "notequal")
    constraint = convert_eq(expr);
  else if (expr.id() == "invalid-pointer")
    constraint = convert_invalid(expr);
  else if (expr.id() == "same-object")
    constraint = convert_same_object(expr);
  else if (expr.id() == "is_dynamic_object")
    constraint = convert_dynamic_object(expr);
  else if (expr.id() == "overflow-+")
    constraint = convert_overflow_sum(expr);
  else if (expr.id() == "overflow--")
    constraint = convert_overflow_sub(expr);
  else if (expr.id() == "overflow-*")
    constraint = convert_overflow_mul(expr);
  else if (expr.id() == "overflow-unary-")
    constraint = convert_overflow_unary(expr);
  else if (has_prefix(expr.id_string(), "overflow-typecast-"))
    constraint = convert_overflow_typecast(expr);
  else if (expr.id() == "member")
    constraint = convert_rest_member(expr);
  else if (expr.id() == "index")
    constraint = convert_rest_index(expr);
  else if (expr.id() == "memory-leak")
    constraint = convert_memory_leak(expr);
#if 0
  else if (expr.id() == "pointer_object_has_type") //@TODO
    ignoring(expr);
#endif
  else if (expr.id() == "is_zero_string") {
    ignoring(expr);
    return l;
  } else
    throw "convert_z3_expr: " + expr.id_string() + " is unsupported";

  formula = Z3_mk_iff(z3_ctx, z3_prop.z3_literal(l), constraint);
  Z3_assert_cnstr(z3_ctx, formula);

  DEBUGLOC;

  if (z3_prop.smtlib)
    z3_prop.assumpt.push_back(formula);

  return l;
}

/*******************************************************************
   Function: z3_convt::convert_rel

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_rel(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);

  Z3_ast operand0, operand1;

  if (convert_bv(expr.op0(), operand0))
    return true;
  if (convert_bv(expr.op1(), operand1))
    return true;

  if (expr.op0().type().id() == "pointer")
    operand0 = z3_api.mk_tuple_select(z3_ctx, operand0, 1);  //select pointer
	                                                     // index

  if (expr.op1().type().id() == "pointer")
    operand1 = z3_api.mk_tuple_select(z3_ctx, operand1, 1);  //select pointer
	                                                     // index

  const typet &op_type = expr.op0().type();

  if (int_encoding) {
    if (expr.id() == "<=")
      bv = Z3_mk_le(z3_ctx, operand0, operand1);
    else if (expr.id() == "<")
      bv = Z3_mk_lt(z3_ctx, operand0, operand1);
    else if (expr.id() == ">=")
      bv = Z3_mk_ge(z3_ctx, operand0, operand1);
    else if (expr.id() == ">")
      bv = Z3_mk_gt(z3_ctx, operand0, operand1);
  } else   {
    if (op_type.id() == "unsignedbv" || op_type.subtype().id() == "unsignedbv"
        || op_type.subtype().subtype().id() == "unsignedbv" ||
        op_type.subtype().id() == "symbol") {
      if (expr.id() == "<=")
	bv = Z3_mk_bvule(z3_ctx, operand0, operand1);
      else if (expr.id() == "<")
	bv = Z3_mk_bvult(z3_ctx, operand0, operand1);
      else if (expr.id() == ">=")
	bv = Z3_mk_bvuge(z3_ctx, operand0, operand1);
      else if (expr.id() == ">")
	bv = Z3_mk_bvugt(z3_ctx, operand0, operand1);
    } else if (op_type.id() == "signedbv" || op_type.id() == "fixedbv" ||
               op_type.id() == "pointer" ||
               op_type.subtype().id() == "signedbv" ||
               op_type.subtype().id() == "fixedbv" ||
               op_type.subtype().id() == "pointer" ||
               op_type.subtype().subtype().id() == "signedbv" ||
               op_type.subtype().subtype().id() == "fixedbv") {
      if (expr.id() == "<=")
	bv = Z3_mk_bvsle(z3_ctx, operand0, operand1);
      else if (expr.id() == "<")
	bv = Z3_mk_bvslt(z3_ctx, operand0, operand1);
      else if (expr.id() == ">=")
	bv = Z3_mk_bvsge(z3_ctx, operand0, operand1);
      else if (expr.id() == ">")
	bv = Z3_mk_bvsgt(z3_ctx, operand0, operand1);
    }
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_typecast

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_typecast_bool(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast args[2];

  if (op.type().id() == "signedbv" ||
      op.type().id() == "unsignedbv" ||
      op.type().id() == "pointer") {
    args[0] = bv;
    if (int_encoding)
      args[1] = z3_api.mk_int(z3_ctx, 0);
    else
      args[1] =
        Z3_mk_int(z3_ctx, 0, Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));

    bv = Z3_mk_distinct(z3_ctx, 2, args);
  } else
    throw "TODO typecast1 " + op.type().id_string() + " -> bool";
}


bool
z3_convt::convert_typecast_fixedbv_nonint(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast args[2];

  const fixedbv_typet &fixedbv_type = to_fixedbv_type(expr.type());
  unsigned to_fraction_bits = fixedbv_type.get_fraction_bits();
  unsigned to_integer_bits = fixedbv_type.get_integer_bits();

  if (op.type().id() == "unsignedbv" ||
      op.type().id() == "signedbv" ||
      op.type().id() == "enum") {
    unsigned from_width;

    if (boolbv_get_width(op.type(), from_width))
      return true;

    if (from_width == to_integer_bits) {
      if (convert_bv(op, bv))
	return true;
    } else if (from_width > to_integer_bits)      {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      bv = Z3_mk_extract(z3_ctx, (from_width - 1), to_integer_bits, args[0]);
    } else   {
      assert(from_width < to_integer_bits);
      if (expr.type().id() == "unsignedbv") {
	if (convert_bv(op, args[0]))
	  return true;

	if (op.type().id() == "pointer")
	  args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

	bv = Z3_mk_zero_ext(z3_ctx, (to_integer_bits - from_width), args[0]);
      } else   {
	if (convert_bv(op, args[0]))
	  return true;

	if (op.type().id() == "pointer")
	  args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

	bv = Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_width), args[0]);
      }
    }

    bv = Z3_mk_concat(z3_ctx, bv, convert_number(0, to_fraction_bits, true));
  } else if (op.type().id() == "bool")      {
    Z3_ast zero, one;
    unsigned width;

    if (boolbv_get_width(expr.type(), width))
      return true;

    zero = convert_number(0, to_integer_bits, true);
    one =  convert_number(1, to_integer_bits, true);
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
    bv = Z3_mk_concat(z3_ctx, bv, convert_number(0, to_fraction_bits, true));
  } else if (op.type().id() == "fixedbv")      {
    Z3_ast magnitude, fraction;
    const fixedbv_typet &from_fixedbv_type = to_fixedbv_type(op.type());
    unsigned from_fraction_bits = from_fixedbv_type.get_fraction_bits();
    unsigned from_integer_bits = from_fixedbv_type.get_integer_bits();
    unsigned from_width = from_fixedbv_type.get_width();

    if (to_integer_bits <= from_integer_bits) {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      magnitude =
        Z3_mk_extract(z3_ctx, (from_fraction_bits + to_integer_bits - 1),
                      from_fraction_bits, args[0]);
    } else   {
      assert(to_integer_bits > from_integer_bits);

      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      magnitude =
        Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_integer_bits),
                       Z3_mk_extract(z3_ctx, from_width - 1, from_fraction_bits,
                                     args[0]));
    }

    if (to_fraction_bits <= from_fraction_bits) {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      fraction =
        Z3_mk_extract(z3_ctx, (from_fraction_bits - 1),
                      from_fraction_bits - to_fraction_bits,
                      args[0]);
    } else   {
      assert(to_fraction_bits > from_fraction_bits);
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      fraction =
        Z3_mk_concat(z3_ctx,
                     Z3_mk_extract(z3_ctx, (from_fraction_bits - 1), 0,args[0]),
                     convert_number(0, to_fraction_bits - from_fraction_bits,
                                    true));
    }
    bv = Z3_mk_concat(z3_ctx, magnitude, fraction);
  } else
    throw "unexpected typecast to fixedbv";

  return false;
}

bool
z3_convt::convert_typecast_ints_ptrs(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast args[2];
  unsigned to_width;

  // Fetch width of target (?) data item
  if (expr.type().id() == "pointer") {
    if (expr.type().subtype().id() == "symbol") {
      to_width = config.ansi_c.int_width;
    } else if (expr.type().subtype().id() == "empty")   {
      to_width = config.ansi_c.int_width;
    } else {
      if (boolbv_get_width(expr.type().subtype(), to_width))
	return true;
    }
  } else   {
    if (boolbv_get_width(expr.type(), to_width))
      return true;
  }

  if (op.type().id() == "signedbv" || op.type().id() == "c_enum" ||
      op.type().id() == "fixedbv" ||
      op.type().subtype().id() == "signedbv" || op.type().subtype().id() ==
      "fixedbv") {
    unsigned from_width;

    if (op.type().id() == "pointer") {
      if (boolbv_get_width(op.type().subtype(), from_width))
	return true;
    } else   {
      if (boolbv_get_width(op.type(), from_width))
	return true;
    }

    if (from_width == to_width) {
      if (convert_bv(op, bv))
	return true;

      if (op.type().id() == "pointer")
	bv = z3_api.mk_tuple_select(z3_ctx, bv, 0);
      else if (int_encoding && op.type().id() == "signedbv" &&
               expr.type().id() == "fixedbv")
	bv = Z3_mk_int2real(z3_ctx, bv);
      else if (int_encoding && op.type().id() == "fixedbv" &&
               expr.type().id() == "signedbv")
	bv = Z3_mk_real2int(z3_ctx, bv);
      // XXXjmorse - there isn't a case here for if !int_encoding

    } else if (from_width < to_width)      {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer") {
	to_width = (to_width == 40) ? config.ansi_c.int_width : to_width;
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);
      }

      if (int_encoding &&
          ((expr.type().id() == "fixedbv" && op.type().id() == "signedbv") ||
           (op.type().id() == "pointer" &&
            expr.type().subtype().id() == "fixedbv")))
	bv = Z3_mk_int2real(z3_ctx, args[0]);
      else if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_sign_ext(z3_ctx, (to_width - from_width), args[0]);
    } else if (from_width > to_width)     {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      if (int_encoding &&
          ((op.type().id() == "signedbv" && expr.type().id() == "fixedbv") /*||
			                                                      (op.type().id()=="pointer"
			                                                         &&
			                                                         op.type().subtype().id()=="fixedbv"))*/))
	bv = Z3_mk_int2real(z3_ctx, args[0]);
      else if (int_encoding && op.type().id() == "pointer" &&
               op.type().subtype().id() == "fixedbv")
	bv = Z3_mk_real2int(z3_ctx, args[0]);
      else if (int_encoding &&
               (op.type().id() == "fixedbv" && expr.type().id() == "signedbv"))
	bv = Z3_mk_real2int(z3_ctx, args[0]);
      else if (int_encoding)
	bv = args[0];
      else {
	if (!to_width) to_width = config.ansi_c.int_width;
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, args[0]);
      }
    }
  } else if (op.type().id() == "unsignedbv" || op.type().subtype().id() ==
             "unsignedbv")       {
    unsigned from_width;

    if (op.type().id() == "pointer") {
      if (boolbv_get_width(op.type().subtype(), from_width))
	return true;
    } else   {
      if (boolbv_get_width(op.type(), from_width))
	return true;
    }

    if (from_width == to_width) {
      if (convert_bv(op, bv))
	return true;

      if (op.type().id() == "pointer")
	bv = z3_api.mk_tuple_select(z3_ctx, bv, 0);
    } else if (from_width < to_width)      {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_zero_ext(z3_ctx, (to_width - from_width), args[0]);
    } else if (from_width > to_width)     {
      if (convert_bv(op, args[0]))
	return true;

      if (op.type().id() == "pointer")
	args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, args[0]);
    }
  } else if (op.type().id() == "bool")     {
    Z3_ast zero = 0, one = 0;
    unsigned width;

    if (boolbv_get_width(expr.type(), width))
      return true;

    if (expr.type().id() == "signedbv") {
      if (int_encoding) {
	zero = Z3_mk_int(z3_ctx, 0, Z3_mk_int_type(z3_ctx));
	one = Z3_mk_int(z3_ctx, 1, Z3_mk_int_type(z3_ctx));
      } else   {
	zero = convert_number(0, width, true);
	one =  convert_number(1, width, true);
      }
    } else if (expr.type().id() == "unsignedbv")     {
      if (int_encoding) {
	zero = Z3_mk_int(z3_ctx, 0, Z3_mk_int_type(z3_ctx));
	one = Z3_mk_int(z3_ctx, 1, Z3_mk_int_type(z3_ctx));
      } else   {
	zero = convert_number(0, width, false);
	one =  convert_number(1, width, false);
      }
    } else if (expr.type().id() == "fixedbv")     {
      if (int_encoding) {
	zero = Z3_mk_int(z3_ctx, 0, Z3_mk_real_type(z3_ctx));
	one = Z3_mk_int(z3_ctx, 1, Z3_mk_real_type(z3_ctx));
      } else   {
	zero = convert_number(0, width, true);
	one =  convert_number(1, width, true);
      }
    }
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
  } else if (op.type().subtype().id() == "empty")      {
    unsigned from_width = config.ansi_c.int_width;
    Z3_ast object, pointer_var;
    Z3_symbol mk_tuple_name, proj_names[2];
    Z3_type_ast proj_types[2], type_var;
    Z3_const_decl_ast mk_tuple_decl, proj_decls[2];

    proj_names[0] = Z3_mk_string_symbol(z3_ctx, "object");

    // XXXjmorse - is this supposed to extract the bit width from the first
    // non-pointer part of the expr type? It's not going to work on an arbitary
    // depth of pointers.
    if (expr.type().subtype().id() != "pointer") {
      if (boolbv_get_width(expr.type().subtype() /*expr.type()*/, to_width))
	to_width = config.ansi_c.int_width;
    } else   {
      if (boolbv_get_width(expr.type().subtype().subtype() /*expr.type()*/,
                           to_width))
	to_width = config.ansi_c.int_width;
    }

    if (int_encoding)
      proj_types[0] = Z3_mk_int_type(z3_ctx);
    else
      proj_types[0] = Z3_mk_bv_type(z3_ctx, to_width);

    char val[3];
    std::string name;
    std::stringstream s;
    sprintf(val, "%i", to_width);
    name = "pointer_tuple_";
    name += Z3_get_symbol_string(z3_ctx, Z3_get_type_name(z3_ctx, proj_types[0]));
    name += val;
    mk_tuple_name = Z3_mk_string_symbol(z3_ctx, name.c_str());

    proj_names[1] = Z3_mk_string_symbol(z3_ctx, "index");

    if (int_encoding)
      proj_types[1] = Z3_mk_int_type(z3_ctx);
    else
      proj_types[1] = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);

    if (from_width == to_width) {
      if (convert_bv(op, args[0]))
	return true;

      if (op.operands().size() == 0
          && op.id() == "constant"
          && op.type().subtype().id() == "empty") {
	return true;
      } else if (op.operands().size() > 0)     {
	if (op.op0().id() == "address_of")
	  bv = z3_api.mk_tuple_select(z3_ctx, args[0], 0);
      }

      return false;
    } else if (from_width < to_width)      {
      if (convert_bv(op, args[0]))
	return true;

      object = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      if (int_encoding)
	bv = object;
      else
	bv = Z3_mk_sign_ext(z3_ctx, (to_width - from_width), object);
    } else if (from_width > to_width)     {
      if ( convert_bv(op, args[0]))
	return true;

      object = z3_api.mk_tuple_select(z3_ctx, args[0], 0);

      if (int_encoding)
	bv = object;
      else
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, object);
    }

    type_var =
      Z3_mk_tuple_type(z3_ctx, mk_tuple_name, 2, proj_names, proj_types,
                       &mk_tuple_decl, proj_decls);
    pointer_var = z3_api.mk_var(z3_ctx, expr.op0().get_string(
                                  "identifier").c_str(), type_var);

    bv = z3_api.mk_tuple_update(z3_ctx, pointer_var, 0, bv);
    bv = z3_api.mk_tuple_update(z3_ctx, pointer_var, 1,
                                z3_api.mk_tuple_select(z3_ctx, args[0], 1));

    return false;
  } else if (op.type().subtype().id() == "symbol" ||
             op.type().subtype().id() == "code")        {
    if (convert_bv(op, args[0])) {
      assert(0);
      return true;
    }

    if (op.id() == "constant") {
      if (op.get("value").compare("NULL") == 0)
	bv = convert_number(0, config.ansi_c.int_width, true);
    } else if (op.operands().size() == 0)     {
      if (expr.type().subtype().id() != "empty" &&
          (op.type().id() == "pointer" && op.type().subtype().id() == "symbol"))
	bv = z3_api.mk_tuple_select(z3_ctx, args[0], 0);
    } else if (expr.type().subtype().id() != "empty" &&
               (op.type().id() == "pointer" && op.type().subtype().id() ==
                "symbol") &&
               op.op0().id() != "index" && expr.type().id() != "pointer"
               /*&& op.op0().type().id()!="struct"*/) {
      bv = z3_api.mk_tuple_select(z3_ctx, args[0], 0);
    } else if (op.id() == "typecast" || op.id() == "member")   {
      bv = z3_api.mk_tuple_select(z3_ctx, args[0], 0);
    }
    return false;
  } else if (op.type().subtype().id() == "pointer")      {
    if (convert_bv(op, bv))
      return true;

    if (op.type().subtype().subtype().id() != "empty" &&
        expr.type().id() != "pointer")
      bv = z3_api.mk_tuple_select(z3_ctx, bv, 0);

    return false;
  } else   {
    return true;
  }
  if (expr.type().id() == "pointer") {
    Z3_ast pointer_var;

    if (convert_z3_pointer(expr, "pointer", pointer_var))
      return true;

    bv = z3_api.mk_tuple_update(z3_ctx, pointer_var, 0, bv);
  }

  return false;
}

bool
z3_convt::convert_typecast_struct(const exprt &expr, Z3_ast &bv)
{

  const struct_typet &struct_type = to_struct_type(expr.op0().type());
  const struct_typet::componentst &components = struct_type.components();
  const struct_typet &struct_type2 = to_struct_type(expr.type());
  const struct_typet::componentst &components2 = struct_type2.components();
  struct_typet s;
  Z3_ast operand;
  u_int i = 0, i2 = 0, j = 0;

  s.components().resize(struct_type2.components().size());

  for (struct_typet::componentst::const_iterator
       it2 = components2.begin();
       it2 != components2.end();
       it2++, i2++)
  {
    for (struct_typet::componentst::const_iterator
         it = components.begin();
         it != components.end();
         it++, i++)
    {
      if (it->get("name").compare(it2->get("name")) == 0) {
	unsigned width;
	if (boolbv_get_width(it->type(), width))
	  return true;
	if (it->type().id() == "signedbv") {
	  s.components()[j].set_name(it->get("name"));
	  s.components()[j].type() = signedbv_typet(width);
	} else if (it->type().id() == "unsignedbv")     {
	  s.components()[j].set_name(it->get("name"));
	  s.components()[j].type() = unsignedbv_typet(width);
	} else if (it->type().id() == "bool")     {
	  s.components()[j].set_name(it->get("name"));
	  s.components()[j].type() = bool_typet();
	} else   {
	  return true;
	}
	j++;
      }
    }
  }
  exprt new_struct("symbol", s);
  new_struct.type().set("tag", expr.type().get_string("tag"));
  new_struct.set("identifier", "typecast_" + expr.op0().get_string("identifier"));

  if (convert_bv(new_struct, operand))
    return true;

  i2 = 0;
  for (struct_typet::componentst::const_iterator
       it2 = components2.begin();
       it2 != components2.end();
       it2++, i2++)
  {
    Z3_ast formula;
    formula = Z3_mk_eq(z3_ctx, z3_api.mk_tuple_select(z3_ctx, operand, i2),
                       z3_api.mk_tuple_select(z3_ctx, bv, i2));
    Z3_assert_cnstr(z3_ctx, formula);
    if (z3_prop.smtlib)
      z3_prop.assumpt.push_back(formula);
  }

  bv = operand;
  return false;
}

bool
z3_convt::convert_typecast_enum(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast zero, one;
  unsigned width;

  if (op.type().id() == "bool") {
    if (boolbv_get_width(expr.type(), width))
      return true;

    zero = convert_number(0, width, true);
    one =  convert_number(1, width, true);
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
  }

  return false;
}

bool
z3_convt::convert_typecast(const exprt &expr, Z3_ast &bv)
{
  //@TODO: Break this method into smaller methods
  assert(expr.operands().size() == 1);
  const exprt &op = expr.op0();

  Z3_ast args[2];

  if (convert_bv(op, bv))
    return true;

  if (expr.type().id() == "bool") {
    return convert_typecast_bool(expr, bv);
  } else if (expr.type().id() == "fixedbv" && !int_encoding)      {
    return convert_typecast_fixedbv_nonint(expr, bv);
  } else if ((expr.type().id() == "signedbv" || expr.type().id() ==
              "unsignedbv"
              || expr.type().id() == "fixedbv" || expr.type().id() ==
              "pointer")) {
    return convert_typecast_ints_ptrs(expr, bv);
  } else if (expr.type().id() == "struct")     {
    return convert_typecast_struct(expr, bv);
  } else if (expr.type().id() == "c_enum")      {
    return convert_typecast_enum(expr, bv);
  }

  // XXXjmorse -- what about all other types, eh?

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_struct

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_struct(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  Z3_ast value;

  const struct_typet &struct_type = to_struct_type(expr.type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0;
  std::string identifier;

  assert(components.size() >= expr.operands().size());
  assert(!components.empty());

  identifier = "conv_struct_" + expr.type().get_string("tag");

  if (convert_identifier(identifier, expr.type(), bv))
    return true;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (convert_bv(expr.operands()[i], value))
      return true;
    bv = z3_api.mk_tuple_update(z3_ctx, bv, i, value);
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_union

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_union(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  Z3_ast value;

  const struct_typet &struct_type = to_struct_type(expr.type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0;

  assert(components.size() >= expr.operands().size());
  assert(!components.empty());

  if (convert_identifier(expr.type().get_string("tag"), expr.type(), bv))
    return true;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {

    if (expr.type().get_string("tag").find("__align'") != std::string::npos &&
        expr.type().id() == "union" && i == 1)
      return false;

    if (convert_bv(expr.operands()[i], value))
      return true;

    bv = z3_api.mk_tuple_update(z3_ctx, bv, i, value);
    bv = z3_api.mk_tuple_update(z3_ctx, bv, components.size(),
                                convert_number(i, config.ansi_c.int_width, 0));
                                //record the last element written to the union

  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_z3_pointer

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_z3_pointer(const exprt &expr, std::string symbol, Z3_ast &bv)
{
  DEBUGLOC;

  Z3_type_ast tuple_type;
  std::string cte, identifier;
  unsigned width;
  char val[2];

  if (check_all_types(expr.type().subtype())) {
    if (expr.type().subtype().id() == "pointer") {
      if (boolbv_get_width(expr.type().subtype().subtype(), width)) {
	if (expr.type().subtype().subtype().id() == "empty"
	    || expr.type().subtype().subtype().id() == "symbol")
	  width = config.ansi_c.int_width;
	else
	  return true;
      }
    } else   {
      if (boolbv_get_width(expr.type().subtype(), width))
	width = config.ansi_c.int_width;
    }

    if (expr.type().subtype().id() == "code") {
      if (create_pointer_type(expr.type(), tuple_type))
	return true;
    } else if (create_pointer_type(expr.type().subtype(), tuple_type))
      return true;

  } else if (check_all_types(expr.type()))   {
    if (boolbv_get_width(expr.type(), width))
      return true;

    if (create_pointer_type(expr.type(), tuple_type))
      return true;
  }

  sprintf(val, "%i", width);
  identifier = symbol;
  identifier += val;

  if (expr.type().subtype().id() == "empty")
    identifier += "empty";

  bv = z3_api.mk_var(z3_ctx, identifier.c_str(), tuple_type);

  if (expr.get("value").compare("NULL") == 0) {
    if (int_encoding)
      bv = z3_api.mk_tuple_update(z3_ctx, bv, 1, z3_api.mk_int(z3_ctx, -1));
    else
      bv =
        z3_api.mk_tuple_update(z3_ctx, bv, 1,
                               Z3_mk_int(z3_ctx, -1,
                                         Z3_mk_bv_type(z3_ctx,
                                                       config.ansi_c.int_width)));
  } else   {
    if (int_encoding)
      bv = z3_api.mk_tuple_update(z3_ctx, bv, 1, z3_api.mk_int(z3_ctx, 0));
    else
      bv =
        z3_api.mk_tuple_update(z3_ctx, bv, 1,
                               Z3_mk_int(z3_ctx, 0,
                                         Z3_mk_bv_type(z3_ctx,
                                                       config.ansi_c.int_width)));
  }

  unsigned object = pointer_logic.add_object(expr);

  if (object && expr.type().subtype().id() == "code") {
    if (int_encoding)
      bv = z3_api.mk_tuple_update(z3_ctx, bv, 0, z3_api.mk_int(z3_ctx, object));
    else
      bv =
        z3_api.mk_tuple_update(z3_ctx, bv, 0,
                               Z3_mk_int(z3_ctx, object,
                                         Z3_mk_bv_type(z3_ctx,
                                                       config.ansi_c.int_width)));
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_zero_string

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_zero_string(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  Z3_type_ast array_type;

  if (create_array_type(expr.type(), array_type))
    return true;

  bv = z3_api.mk_var(z3_ctx, "zero_string", array_type);

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_array

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_array(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  u_int width = 0, i = 0;
  Z3_type_ast array_type, tuple_type;
  Z3_ast array_cte, int_cte, val_cte, tmp_struct;
  std::string value_cte;
  char i_str[2];

  width = config.ansi_c.int_width;

  if (expr.type().subtype().id() == "fixedbv") {
    if (boolbv_get_width(expr.type().subtype(), width))
      return true;

    if (int_encoding)
      array_type  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(
                                       z3_ctx), Z3_mk_real_type(z3_ctx));
    else
      array_type  =
        Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                                               config.ansi_c.int_width),
                         Z3_mk_bv_type(z3_ctx, width));
  } else if (expr.type().subtype().id() == "struct")   {
    if (create_struct_type(expr.op0().type(), tuple_type))
      return true;

    if (int_encoding)
      array_type  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(z3_ctx), tuple_type);
    else
      array_type =
        Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                                               config.ansi_c.int_width),
                         tuple_type);

    value_cte = "constant" + expr.op0().type().get_string("tag");
    array_cte = z3_api.mk_var(z3_ctx, value_cte.c_str(), array_type);

    i = 0;
    forall_operands(it, expr)
    {
      sprintf(i_str, "%i", i);
      if (int_encoding)
	int_cte = z3_api.mk_int(z3_ctx, atoi(i_str));
      else
	int_cte =
	  Z3_mk_int(z3_ctx, atoi(i_str),
	            Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));

      if (convert_struct(*it, tmp_struct))
	return true;

      array_cte = Z3_mk_store(z3_ctx, array_cte, int_cte, tmp_struct);
      ++i;
    }

    bv = array_cte;


    DEBUGLOC;

    return false;
    //return array_cte;
  } else if (expr.type().subtype().id() == "array")   {
    if (boolbv_get_width(expr.type().subtype().subtype(), width))
      return true;

    if (create_array_type(expr.type(), array_type))
      return true;
  } else   {
	DEBUGLOC;

    if (boolbv_get_width(expr.type().subtype(), width))
      return true;

    if (expr.type().subtype().subtype().id()=="code")
    	  width=config.ansi_c.int_width;

    if (expr.type().subtype().id()=="pointer") {
    	assert(width==config.ansi_c.int_width);
    }

    if (int_encoding)
      array_type  = Z3_mk_array_type(z3_ctx, Z3_mk_int_type(
                                       z3_ctx), Z3_mk_int_type(z3_ctx));
    else
      array_type  =
        Z3_mk_array_type(z3_ctx, Z3_mk_bv_type(z3_ctx,
                                               config.ansi_c.int_width),
                         Z3_mk_bv_type(z3_ctx, width));
  }

  value_cte = expr.get_string("identifier") +
              expr.type().subtype().get("width").c_str();
  bv = z3_api.mk_var(z3_ctx, value_cte.c_str(), array_type);

  i = 0;
  forall_operands(it, expr)
  {
    sprintf(i_str, "%i", i);
    if (int_encoding)
      int_cte = z3_api.mk_int(z3_ctx, atoi(i_str));
    else
      int_cte =
        Z3_mk_int(z3_ctx, atoi(i_str),
                  Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));

    if (it->type().id() == "array") {
      if (convert_bv(*it, val_cte))
	return true;
    } else   {
      if (is_signed(it->type()))
	value_cte =
	  integer2string(binary2integer(it->get("value").c_str(), true), 10);
      else
	value_cte =
	  integer2string(binary2integer(it->get("value").c_str(), false), 10);
      if (int_encoding) {
	    if (it->type().id() == "signedbv" || it->type().id() == "pointer") {
	      val_cte =
	    		  Z3_mk_int(z3_ctx, atoi(value_cte.c_str()), Z3_mk_int_type(z3_ctx));
	    } else if (it->type().id() == "unsignedbv") {
	    	val_cte = Z3_mk_unsigned_int(z3_ctx, atoi(
	                                 value_cte.c_str()),
	                               Z3_mk_int_type(z3_ctx));
	    } else if (it->type().id() == "fixedbv") {
	    	val_cte =
	    			Z3_mk_int(z3_ctx, atoi(value_cte.c_str()), Z3_mk_real_type(z3_ctx));
	    }
      } else   {
    	  DEBUGLOC;
	val_cte =
	  Z3_mk_int(z3_ctx, atoi(value_cte.c_str()),
	            Z3_mk_bv_type(z3_ctx, width));
      }
    }

    bv = Z3_mk_store(z3_ctx, bv, int_cte, val_cte);
    ++i;
  }


  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_constant

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_constant(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  std::string value;
  unsigned width;

  if (expr.type().id() == "c_enum") {
    // jmorse: value field of C enum type is in fact base 10, wheras everything
    // else is base 2.
    value = expr.get_string("value");
  } else if (expr.type().id() == "bool")   {
    // value will not actually be interpreted as number by below code
    value = expr.get_string("value");
  } else if (expr.type().id() == "pointer" && expr.get_string("value") ==
             "NULL")   {
    // Uuugghhhh. Match what happens if we were to feed this to binary2integer.
    value = "0";
  } else if (is_signed(expr.type()))   {
    value = integer2string(binary2integer(expr.get_string("value"), true), 10);
  } else {

    value = integer2string(binary2integer(expr.get_string("value"), false), 10);
  }

  if (expr.type().id() == "unsignedbv") {
    if (boolbv_get_width(expr.type(), width))
      return true;

    if (int_encoding)
      bv = z3_api.mk_unsigned_int(z3_ctx, atoi(value.c_str()));
    else
      bv =
        Z3_mk_unsigned_int(z3_ctx, atoi(value.c_str()),
                           Z3_mk_bv_type(z3_ctx, width));
  }
  if (expr.type().id() == "signedbv" || expr.type().id() == "c_enum") {
    if (boolbv_get_width(expr.type(), width))
      return true;

    if (int_encoding)
      bv = z3_api.mk_int(z3_ctx, atoi(value.c_str()));
    else
      bv = Z3_mk_int(z3_ctx, atoi(value.c_str()), Z3_mk_bv_type(z3_ctx, width));
  } else if (expr.type().id() == "fixedbv")    {
    if (boolbv_get_width(expr.type(), width))
      return true;

    if (int_encoding) {
      std::string result;
      result = fixed_point(expr.get_string("value"), width);
      bv = Z3_mk_numeral(z3_ctx, result.c_str(), Z3_mk_real_type(z3_ctx));
    }   else {
      Z3_ast magnitude, fraction;
      std::string m, f, c;
      m = extract_magnitude(expr.get_string("value"), width);
      f = extract_fraction(expr.get_string("value"), width);
      magnitude =
        Z3_mk_int(z3_ctx, atoi(m.c_str()), Z3_mk_bv_type(z3_ctx, width / 2));
      fraction =
        Z3_mk_int(z3_ctx, atoi(f.c_str()), Z3_mk_bv_type(z3_ctx, width / 2));
      bv = Z3_mk_concat(z3_ctx, magnitude, fraction);
    }
  } else if (expr.type().id() == "pointer")    {
    if (convert_z3_pointer(expr, value, bv))
      return true;
  } else if (expr.type().id() == "bool")     {
    if (expr.is_false())
      bv = Z3_mk_false(z3_ctx);
    else if (expr.is_true())
      bv = Z3_mk_true(z3_ctx);
  } else if (expr.type().id() == "array")     {
    if (convert_array(expr, bv))
      return true;
  } else if (expr.type().id() == "struct")     {
    if (convert_struct(expr, bv))
      return true;
  } else if (expr.type().id() == "union")     {
    if (convert_union(expr, bv))
      return true;
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_bitwise

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_bitwise(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() >= 2);
  Z3_ast *args;
  unsigned i = 0, width;

  args = new Z3_ast[expr.operands().size() + 1];

  if (convert_bv(expr.op0(), args[0]))
    return true;
  if (convert_bv(expr.op1(), args[1]))
    return true;

  forall_operands(it, expr)
  {
    if (convert_bv(*it, args[i]))
      return true;

    if (int_encoding) {
      if (boolbv_get_width(expr.type(), width))
	return true;

      args[i] = Z3_mk_int2bv(z3_ctx, width, args[i]);
    }

    if (i >= 1) {
      if (expr.id() == "bitand")
	args[i + 1] = Z3_mk_bvand(z3_ctx, args[i - 1], args[i]);
      else if (expr.id() == "bitor")
	args[i + 1] = Z3_mk_bvor(z3_ctx, args[i - 1], args[i]);
      else if (expr.id() == "bitxor")
	args[i + 1] = Z3_mk_bvxor(z3_ctx, args[i - 1], args[i]);
      else if (expr.id() == "bitnand")
	args[i + 1] = Z3_mk_bvnand(z3_ctx, args[i - 1], args[i]);
      else if (expr.id() == "bitnor")
	args[i + 1] = Z3_mk_bvnor(z3_ctx, args[i - 1], args[i]);
      else if (expr.id() == "bitnxor")
	args[i + 1] = Z3_mk_bvxnor(z3_ctx, args[i - 1], args[i]);
    }
    ++i;
  }

  if (int_encoding) {
    if (expr.type().id() == "signedbv")
      bv = Z3_mk_bv2int(z3_ctx, args[i], true);
    else if (expr.type().id() == "unsignedbv")
      bv = Z3_mk_bv2int(z3_ctx, args[i], false);
    else
      throw "bitwise operations are not supported";
  } else
    bv = args[i];

  delete[] args;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_unary_minus

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_unary_minus(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast args[2];

  if (convert_bv(expr.op0(), args[0]))
    return true;

  if (int_encoding) {
    if (expr.type().id() == "signedbv" || expr.type().id() == "unsignedbv") {
      args[1] = z3_api.mk_int(z3_ctx, -1);
      bv = Z3_mk_mul(z3_ctx, 2, args);
    } else if (expr.type().id() == "fixedbv")   {
      args[1] = Z3_mk_int(z3_ctx, -1, Z3_mk_real_type(z3_ctx));
      bv = Z3_mk_mul(z3_ctx, 2, args);
    }
  } else   {
    bv = Z3_mk_bvneg(z3_ctx, args[0]);
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_if

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_if(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 3);

  Z3_ast operand0, operand1, operand2;

  if (convert_bv(expr.op0(), operand0))
    return true;

  if (convert_bv(expr.op1(), operand1))
    return true;

  if (convert_bv(expr.op2(), operand2))
    return true;

  bv = Z3_mk_ite(z3_ctx, operand0, operand1, operand2);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_logical_ops

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_logical_ops(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.type().id() == "bool");
  assert(expr.operands().size() >= 1);

  Z3_ast *args;
  u_int i = 0, size;

  size = expr.operands().size();
  args = new Z3_ast[size];

  if (expr.operands().size() == 1) {
    if (convert_bv(expr.op0(), bv))
      return true;
  } else   {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (i >= 1 && expr.id() == "xor")
	args[i] = Z3_mk_xor(z3_ctx, args[i - 1], args[i]);
      ++i;
    }

    if (expr.id() == "and")
      bv = Z3_mk_and(z3_ctx, i, args);
    else if (expr.id() == "or")
      bv = Z3_mk_or(z3_ctx, i, args);
    else if (expr.id() == "xor")
      bv = args[size - 1];
  }

  delete[] args;

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_logical_not

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_logical_not(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand0;

  if (convert_bv(expr.op0(), operand0))
    return true;

  bv = Z3_mk_not(z3_ctx, operand0);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_equality

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_equality(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  //std::cout << "expr.pretty(): " << expr.pretty() << std::endl;

  assert(expr.operands().size() == 2);
  assert(expr.op0().type() == expr.op1().type());

  Z3_ast args[2];

  if (convert_bv(expr.op0(), args[0]))
    return true;

  if (convert_bv(expr.op1(), args[1]))
    return true;

  if (expr.id() == "=")
    bv = Z3_mk_eq(z3_ctx, args[0], args[1]);
  else
    bv = Z3_mk_distinct(z3_ctx, 2, args);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_add

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_add(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() >= 2);
  Z3_ast *args;
  u_int i = 0, size;

  size = expr.operands().size() + 1;
  args = new Z3_ast[size];

  if (expr.type().id() == "pointer" ||
      expr.op0().type().id() == "pointer" ||
      expr.op1().type().id() == "pointer") {
    Z3_ast pointer = 0;

    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (it->type().id() == "pointer") {
	pointer = args[i];

	args[i] = z3_api.mk_tuple_select(z3_ctx, pointer, 1); //select pointer
				                              // index
      }

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_add(z3_ctx, i, args);

    bv = z3_api.mk_tuple_update(z3_ctx, pointer, 1, args[i]);

    if (expr.type().id() == "signedbv")
      bv = args[i];
  } else if (expr.op0().id() == "typecast" || expr.op1().id() ==
             "typecast")       {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (it->id() == "typecast") {
	const exprt &offset = it->operands()[0];

	if (offset.type().id() == "pointer" && offset.type().subtype().id() !=
	    "empty" /*&& offset.type().subtype().id()!="signedbv"*/)
	  args[i] = z3_api.mk_tuple_select(z3_ctx, args[i], 1);     //select
				                                    // pointer
				                                    // index
      }

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_add(z3_ctx, i, args);

    bv = args[i];
  } else   {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvadd(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_add(z3_ctx, i, args);

    bv = args[i];
  }

  delete[] args;

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_sub

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_sub(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() >= 2);
  Z3_ast *args;
  u_int i = 0, size;

  size = expr.operands().size() + 1;
  args = new Z3_ast[size];

  if (expr.op0().type().id() == "pointer" || expr.op1().type().id() ==
      "pointer") {
    Z3_ast pointer = 0;

    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (it->type().id() == "pointer") {
	pointer = args[i];
	args[i] = z3_api.mk_tuple_select(z3_ctx, pointer, 1); //select pointer
				                              // index
      }

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvsub(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvsub(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_sub(z3_ctx, i, args);

    bv = z3_api.mk_tuple_update(z3_ctx, pointer, 1, args[i]);

    if (expr.type().id() == "signedbv")
      bv = args[i];
  }
  else {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvsub(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvsub(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_sub(z3_ctx, i, args);

    bv = args[i];
  }

  delete[] args;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_div

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_div(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;

  if (convert_bv(expr.op0(), operand0))
    return true;
  if (convert_bv(expr.op1(), operand1))
    return true;

  if (int_encoding) {
    bv = Z3_mk_div(z3_ctx, operand0, operand1);
  } else   {
    if (expr.type().id() == "signedbv")
      bv = Z3_mk_bvsdiv(z3_ctx, operand0, operand1);
    else if (expr.type().id() == "unsignedbv")
      bv = Z3_mk_bvudiv(z3_ctx, operand0, operand1);
    else if (expr.type().id() == "fixedbv") {
      fixedbvt fbt(expr);
      unsigned fraction_bits = fbt.spec.get_fraction_bits();

      bv = Z3_mk_extract(z3_ctx, fbt.spec.width - 1, 0,
                         Z3_mk_bvsdiv(z3_ctx,
                                      Z3_mk_concat(z3_ctx, operand0,
                                                   convert_number(0,
                                                                  fraction_bits, true)),
                                      Z3_mk_sign_ext(z3_ctx, fraction_bits,
                                                     operand1)));
    } else
      throw "unsupported type for /: " + expr.type().id_string();
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_mod

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_mod(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;

  if (convert_bv(expr.op0(), operand0))
    return true;
  if (convert_bv(expr.op1(), operand1))
    return true;

  if (int_encoding) {
    if (expr.type().id() == "signedbv" || expr.type().id() == "unsignedbv")
      bv = Z3_mk_mod(z3_ctx, operand0, operand1);
    else if (expr.type().id() == "fixedbv")
      throw "unsupported type for mod: " + expr.type().id_string();
  } else   {
    if (expr.type().id() == "signedbv") {
      bv = Z3_mk_bvsrem(z3_ctx, operand0, operand1);
    }   else if (expr.type().id() == "unsignedbv") {
      bv = Z3_mk_bvurem(z3_ctx, operand0, operand1);
    } else if (expr.type().id() == "fixedbv")
      throw "unsupported type for mod: " + expr.type().id_string();
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_mul

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_mul(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() >= 2);
  Z3_ast *args;
  u_int i = 0, size;
  unsigned fraction_bits = 0;
  size = expr.operands().size() + 1;
  args = new Z3_ast[size];

  if (expr.type().id() == "fixedbv") {
    fixedbvt fbt(expr);
    fraction_bits = fbt.spec.get_fraction_bits();
  }

  if (expr.op0().type().id() == "pointer" || expr.op1().type().id() ==
      "pointer") {
    Z3_ast pointer = 0;

    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (it->type().id() == "pointer") {
	pointer = args[i];
	args[i] = z3_api.mk_tuple_select(z3_ctx, pointer, 1); //select pointer
				                              // index
      }

      if (expr.type().id() == "fixedbv" && !int_encoding)
	args[i] = Z3_mk_sign_ext(z3_ctx, fraction_bits, args[i]);

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_mul(z3_ctx, i, args);

    bv = z3_api.mk_tuple_update(z3_ctx, pointer, 1, args[i]);
  } else if (expr.op0().id() == "typecast" || expr.op1().id() ==
             "typecast")       {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (it->id() == "typecast") {
	const exprt &offset = it->operands()[0];
	if (offset.type().id() == "pointer")
	  args[i] = z3_api.mk_tuple_select(z3_ctx, args[i], 1);     //select
				                                    // pointer
				                                    // index
      }

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_mul(z3_ctx, i, args);

    bv = args[i];
  } else   {
    forall_operands(it, expr)
    {
      if (convert_bv(*it, args[i]))
	return true;

      if (expr.type().id() == "fixedbv" && !int_encoding)
	args[i] = Z3_mk_sign_ext(z3_ctx, fraction_bits, args[i]);

      if (!int_encoding) {
	if (i == 1) {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[0], args[1]);
	} else if (i > 1)     {
	  args[size - 1] = Z3_mk_bvmul(z3_ctx, args[size - 1], args[i]);
	}
      }
      ++i;
    }

    if (int_encoding)
      args[i] = Z3_mk_mul(z3_ctx, i, args);

    bv = args[i];
  }

  if (expr.type().id() == "fixedbv" && !int_encoding) {
    fixedbvt fbt(expr);
    bv =
      Z3_mk_extract(z3_ctx, fbt.spec.width + fraction_bits - 1, fraction_bits,
                    bv);
  }

  delete[] args;

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_pointer

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_pointer(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  assert(expr.type().id() == "pointer");

  Z3_ast pointer_var, pointer;
  Z3_type_ast pointer_type;
  Z3_ast offset, po, pi;
  std::string symbol_name, out;

  if (create_pointer_type(expr.type(), pointer_type))
    return true;

  offset = convert_number(0, config.ansi_c.int_width, true);

  if (expr.id() == "symbol" ||
      expr.id() == "constant" ||
      expr.id() == "string-constant") {
    pointer_logic.add_object(expr);
  }

  if (expr.op0().id() == "index") {
    const exprt &object = expr.op0().operands()[0];
    const exprt &index = expr.op0().operands()[1];

    symbol_name = "address_of_index" + object.id_string() + object.get_string(
      "identifier");
    pointer_var = z3_api.mk_var(z3_ctx, symbol_name.c_str(), pointer_type);

    if (object.id() == "zero_string") {
      if (convert_zero_string(object, po))
	return true;
    } else if (object.id() == "string-constant")     {
      if (convert_bv(object, po))
	return true;
    } else if (object.type().id() == "array" && object.id() != "member"
               && object.type().subtype().id() != "struct") {
      if (read_cache(object, po))
	return true;
    } else   {
      if (convert_bv(object, po))
	return true;
    }

    if (convert_bv(index, pi))
      return true;

    if (select_pointer_value(po, pi, pointer))
      return true;

    if (expr.op0().type().id() != "struct" && expr.op0().type().id() !=
        "union") {

      pointer_var = z3_api.mk_tuple_update(z3_ctx, pointer_var, 0, pointer); //update
			                                                     // object
      bv = z3_api.mk_tuple_update(z3_ctx, pointer_var, 1, pi); //update offset
      return false;
    }

  } else if (expr.op0().id() == "symbol")   {

    if (expr.op0().type().id() == "signedbv" || expr.op0().type().id() ==
        "fixedbv"
        || expr.op0().type().id() == "unsignedbv") {
      if (convert_z3_pointer(expr, expr.op0().get_string("identifier"),
                             pointer_var))
	return true;
    } else if (expr.op0().type().id() == "pointer")   {
      if (convert_bv(expr.op0(), pointer_var))
	return true;
    } else if (expr.op0().type().id() == "bool")   {
      if (convert_z3_pointer(expr, expr.op0().get_string("identifier"),
                             pointer_var))
	return true;
    } else if (expr.op0().type().id() == "struct"
               || expr.op0().type().id() == "union") {

      char val[2];
      static int count = 0;
      std::string identifier;
      sprintf(val, "%i", count++);
      identifier = "address_of_struct_";
      identifier += val;

      if (expr.op0().type().id() == "struct")
	symbol_name = identifier + expr.op0().get_string("identifier");
      else
	symbol_name = "address_of_union" + expr.op0().get_string("identifier");

      pointer_var = z3_api.mk_var(z3_ctx, symbol_name.c_str(), pointer_type);

      if (convert_bv(expr.op0(), pointer))
	return true;

      if (expr.type().subtype().id() == "symbol"
          && expr.op0().get_string("identifier").find("symex_dynamic") ==
          std::string::npos) {
	pointer = z3_api.mk_tuple_select(z3_ctx, pointer, 0);
      }

      if (expr.type().subtype().id() != "struct") {
	const struct_typet &struct_type = to_struct_type(expr.op0().type());
	const struct_typet::componentst &components = struct_type.components();

	assert(components.size() >= expr.operands().size());
	assert(!components.empty());

	pointer = convert_number(pointer_logic.add_object(
	                           expr), config.ansi_c.int_width, true);

	//show_bv_size(pointer);
      }

      pointer_var = z3_api.mk_tuple_update(z3_ctx, pointer_var, 0, pointer);     //update
			                                                         // object
    }   else if (expr.op0().type().id() == "code") {
      if (convert_z3_pointer(expr, expr.op0().get_string("identifier"),
                             pointer_var))
	return true;
    }
  } else if (expr.op0().id() == "member")   {
    const exprt &object = expr.op0().operands()[0];

    symbol_name = "address_of_member" + object.get_string("identifier");
    pointer_var = z3_api.mk_var(z3_ctx, symbol_name.c_str(), pointer_type);

    if (convert_bv(expr.op0(), pointer))
      return true;

    //workaround
    if (expr.op0().type().get_string("tag").find("__pthread_mutex_s") ==
        std::string::npos) {

      if (expr.op0().type().subtype().id() == "symbol")
	pointer_var =
	  z3_api.mk_tuple_update(z3_ctx, pointer, 0,
	                         convert_number(pointer_logic.add_object(expr),
	                                        config.ansi_c.int_width,
	                                        true));                                                                                              //update
			                                                                                                                             // object
      else if (expr.type().subtype().id() == "symbol")
	pointer_var = z3_api.mk_tuple_update(
	  z3_ctx, pointer_var, 0,
	  convert_number(pointer_logic.add_object(expr),
	                 config.ansi_c.int_width,
	                 true));                                                                                                                         //update
			                                                                                                                                 // object
      else
	pointer_var = z3_api.mk_tuple_update(z3_ctx, pointer_var, 0, pointer);         //update
			                                                               // object
    }
  }

  bv = z3_api.mk_tuple_update(z3_ctx, pointer_var, 1, offset); //update offset

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_array_of

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_array_of(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.type().id() == "array");
  assert(expr.operands().size() == 1);

  Z3_ast value, index;
  Z3_type_ast array_type = 0;
  const array_typet &array_type_size = to_array_type(expr.type());
  std::string tmp, out, identifier;
  static u_int size, j, inc = 0;
  unsigned width;

  tmp =
    integer2string(binary2integer(array_type_size.size().get_string(
                                    "value"), false), 10);
  size = atoi(tmp.c_str());

  size = (size == 0) ? 100 : size; //fill in at least one position

  index = convert_number(j, config.ansi_c.int_width, true);

  if (convert_bv(expr.op0(), value))
    return true;
  if (create_array_type(expr.type(), array_type))
    return true;
  if (boolbv_get_width(expr.op0().type(), width))
    return true;

  if (expr.type().subtype().id() == "bool") {

    value = Z3_mk_false(z3_ctx);
    if (width == 1) out = "width: " + width;
    identifier = "ARRAY_OF(false)" + width;
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  } else if (expr.type().subtype().id() == "signedbv" ||
             expr.type().subtype().id() == "unsignedbv")       {
    ++inc;
    identifier = "ARRAY_OF(0)" + width + inc;
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  } else if (expr.type().subtype().id() == "fixedbv")     {
    identifier = "ARRAY_OF(0l)" + width;
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  } else if (expr.type().subtype().id() == "pointer")     {
    identifier = "ARRAY_OF(0p)" + width;
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
    value = z3_api.mk_tuple_select(z3_ctx, value, 0);
  } else if (expr.type().id() == "array" && expr.type().subtype().id() ==
             "struct")       {
    std::string identifier;
    identifier = "array_of_" + expr.op0().type().get_string("tag");
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  } else if (expr.type().id() == "array" && expr.type().subtype().id() ==
             "union")       {
    std::string identifier;
    identifier = "array_of_" + expr.op0().type().get_string("tag");
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  } else if (expr.type().subtype().id() == "array")     {
    ++inc;
    identifier = "ARRAY_OF(0a)" + width + inc;
    out = "identifier: " + identifier;
    bv = z3_api.mk_var(z3_ctx, identifier.c_str(), array_type);
  }

  //update array
  for (j = 0; j < size; j++)
  {
    index = convert_number(j, config.ansi_c.int_width, true);
    bv = Z3_mk_store(z3_ctx, bv, index, value);
    out = "j: " + j;
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_index

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_index(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);

  Z3_ast args[2], pointer;
  std::string identifier;

  if (convert_bv(expr.op0(), args[0]))
    return true;

  if (convert_bv(expr.op1(), args[1]))
    return true;

  if (expr.op0().type().subtype().id() == "pointer") {

    bv = Z3_mk_select(z3_ctx, args[0], args[1]);

    if (convert_z3_pointer(expr.op0(), "index", pointer))
      return true;

    bv = z3_api.mk_tuple_update(z3_ctx, pointer, 0, bv);
  } else   {
    bv = Z3_mk_select(z3_ctx, args[0], args[1]);
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_shift

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_shift(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;
  unsigned width_expr, width_op0, width_op1;

  if (convert_bv(expr.op0(), operand0))
    return true;
  if (convert_bv(expr.op1(), operand1))
    return true;

  if (boolbv_get_width(expr.type(), width_expr))
    return true;

  if (boolbv_get_width(expr.op0().type(), width_op0))
    return true;

  if (boolbv_get_width(expr.op1().type(), width_op1))
    return true;

  if (int_encoding) {
    operand0 = Z3_mk_int2bv(z3_ctx, width_op0, operand0);
    operand1 = Z3_mk_int2bv(z3_ctx, width_op1, operand1);
  }

  if (width_op0 > width_expr)
    operand0 = Z3_mk_extract(z3_ctx, (width_expr - 1), 0, operand0);
  if (width_op1 > width_expr)
    operand1 = Z3_mk_extract(z3_ctx, (width_expr - 1), 0, operand1);

  if (width_op0 > width_op1) {
    if (expr.op0().type().id() == "unsignedbv")
      operand1 = Z3_mk_zero_ext(z3_ctx, (width_op0 - width_op1), operand1);
    else
      operand1 = Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op1), operand1);
  }

  if (expr.id() == "ashr")
    bv = Z3_mk_bvashr(z3_ctx, operand0, operand1);
  else if (expr.id() == "lshr")
    bv = Z3_mk_bvlshr(z3_ctx, operand0, operand1);
  else if (expr.id() == "shl")
    bv = Z3_mk_bvshl(z3_ctx, operand0, operand1);
  else
    assert(false);

  if (int_encoding) {
    if (expr.type().id() == "signedbv")
      bv = Z3_mk_bv2int(z3_ctx, bv, true);
    else if (expr.type().id() == "unsignedbv")
      bv = Z3_mk_bv2int(z3_ctx, bv, false);
    else
      throw "bitshift operations are not supported";
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_abs

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_abs(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  unsigned width;
  //std::string out;

  if (boolbv_get_width(expr.type(), width))
    return true;

  const exprt::operandst &operands = expr.operands();

  if (operands.size() != 1)
    throw "abs takes one operand";

  const exprt &op0 = expr.op0();
  Z3_ast zero, is_negative, operand[2], val_mul;

  if (expr.type().id() == "signedbv") {
    if (int_encoding) {
      zero = z3_api.mk_int(z3_ctx, 0);
      operand[1] = z3_api.mk_int(z3_ctx, -1);
    } else   {
      zero = convert_number(0, width, true);
      operand[1] = convert_number(-1, width, true);
    }
  } else if (expr.type().id() == "fixedbv")     {
    if (int_encoding) {
      zero = Z3_mk_int(z3_ctx, 0, Z3_mk_real_type(z3_ctx));
      operand[1] = Z3_mk_int(z3_ctx, -1, Z3_mk_real_type(z3_ctx));
    } else   {
      zero = convert_number(0, width, true);
      operand[1] = convert_number(-1, width, true);
    }
  } else if (expr.type().id() == "unsignedbv")     {
    if (int_encoding) {
      zero = z3_api.mk_int(z3_ctx, 0);
      operand[1] = z3_api.mk_int(z3_ctx, -1);
    } else   {
      zero = convert_number(0, width, false);
      operand[1] = convert_number(-1, width, true);
    }
  }

  if (convert_bv(op0, operand[0]))
    return true;

  if (expr.type().id() == "signedbv" || expr.type().id() == "fixedbv") {
    if (int_encoding)
      is_negative = Z3_mk_lt(z3_ctx, operand[0], zero);
    else
      is_negative = Z3_mk_bvslt(z3_ctx, operand[0], zero);
  } else {
    // XXXjmorse - the other case handled in this function is unsignedbv, which
    // is never < 0, so is_negative is always false. However I don't know
    // where the guarentee that only those types come here is.
    is_negative = false;
  }

  if (int_encoding)
    val_mul = Z3_mk_mul(z3_ctx, 2, operand);
  else
    val_mul = Z3_mk_bvmul(z3_ctx, operand[0], operand[1]);

  bv = Z3_mk_ite(z3_ctx, is_negative, val_mul, operand[0]);

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_with

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_with(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 3);
  Z3_ast array_var, array_val, operand0, operand1, operand2;

  if (expr.type().id() == "struct") {
    if (convert_bv(expr.op0(), array_var))
      return true;

    if (convert_bv(expr.op2(), array_val))
      return true;

    bv = z3_api.mk_tuple_update(z3_ctx, array_var,
                                convert_member_name(expr.op0(),
                                                    expr.op1()), array_val);
  } else if (expr.type().id() == "union")   {
    if (convert_bv(expr.op0(), array_var))
      return true;

    if (convert_bv(expr.op2(), array_val))
      return true;

    const struct_typet &struct_type = to_struct_type(expr.op0().type());
    const struct_typet::componentst &components = struct_type.components();

    assert(components.size() >= expr.op0().operands().size());

    bv = z3_api.mk_tuple_update(z3_ctx, array_var,
                                convert_member_name(expr.op0(),
                                                    expr.op1()), array_val);

    bv = z3_api.mk_tuple_update(z3_ctx, bv, components.size(),
                                convert_number(convert_member_name(expr.op0(),
                                                                   expr.op1()),
                                               config.ansi_c.int_width, 0));
  } else   {

    if (convert_bv(expr.op0(), operand0))
      return true;

    if (convert_bv(expr.op1(), operand1))
      return true;

    if (convert_bv(expr.op2(), operand2))
      return true;

    if (expr.op2().type().id() == "pointer") {
      if (select_pointer_value(operand0, operand1, operand2)) //select pointer
				                              // value
	return true;
    }

    bv = Z3_mk_store(z3_ctx, operand0, operand1, operand2);
  }

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_bitnot

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_bitnot(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand0;

  if (convert_bv(expr.op0(), operand0))
    return true;

  if (int_encoding && expr.op0().id() == "member") {
    bv = operand0;
    return false;
  }


  if (int_encoding)
    bv = Z3_mk_not(z3_ctx, operand0);
  else
    bv = Z3_mk_bvnot(z3_ctx, operand0);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_member_name

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

u_int
z3_convt::convert_member_name(const exprt &lhs, const exprt &rhs)
{
  DEBUGLOC;

  const struct_typet &struct_type = to_struct_type(lhs.type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0, resp = 0;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (it->get("name").compare(rhs.get_string("component_name")) == 0)
      resp = i;
  }

  return resp;
}

/*******************************************************************
   Function: z3_convt::convert_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_object(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast args[2];

  if (convert_bv(expr.op0(), args[0]))
    return true;
  if (convert_bv(expr.op1(), args[1]))
    return true;

  //@Lucas
  if (expr.op0().type().id() == "pointer" &&
      expr.op0().type().subtype().id() == "symbol"
      && expr.op1().id() == "address_of") {
    args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 1); //compare the offset
    args[1] = z3_api.mk_tuple_select(z3_ctx, args[1], 1);     //compare the
		                                              // offset
  } else   {
    args[0] = z3_api.mk_tuple_select(z3_ctx, args[0], 0);     //compare the
		                                              // object
    args[1] = z3_api.mk_tuple_select(z3_ctx, args[1], 0);     //compare the
		                                              // object
  }

  bv = Z3_mk_distinct(z3_ctx, 2, args);

  DEBUGLOC;
  return false;
}

/*******************************************************************
   Function: z3_convt::select_pointer_offset

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::select_pointer_offset(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast pointer;

  if (convert_bv(expr.op0(), pointer))
    return true;

  bv = z3_api.mk_tuple_select(z3_ctx, pointer, 1); //select pointer offset

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_member

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_member(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  const struct_typet &struct_type = to_struct_type(expr.op0().type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0, j = 0;
  Z3_ast struct_var;

  if (convert_bv(expr.op0(), struct_var))
    return true;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (it->get("name").compare(expr.get_string("component_name")) == 0)
      j = i;
  }

#if 1
  if (expr.op0().type().id() == "union") {
    union_varst::const_iterator cache_result = union_vars.find(
      expr.op0().get_string("identifier").c_str());
    if (cache_result != union_vars.end()) {
      bv = z3_api.mk_tuple_select(z3_ctx, struct_var, cache_result->second);
      return false;
    }
  }
#endif

  bv = z3_api.mk_tuple_select(z3_ctx, struct_var, j);

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: convert_pointer_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_pointer_object(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1 && is_ptr(expr.op0().type()));
  Z3_ast pointer_object = 0;
  const exprt &object = expr.op0().operands()[0];
  unsigned width, object_width;

  if (boolbv_get_width(expr.type(), width))
    return true;

  if (expr.op0().id() == "symbol" || expr.op0().id() == "constant") {
    if (convert_bv(expr.op0(), pointer_object))
      return true;
    pointer_object = z3_api.mk_tuple_select(z3_ctx, pointer_object, 0);

    if (expr.op0().type().id() == "pointer") {
      if (expr.op0().type().subtype().id() == "empty" ||
          expr.op0().type().subtype().id() == "symbol") {
	object_width = config.ansi_c.int_width;
      } else   {
	if (boolbv_get_width(expr.op0().type().subtype(), object_width))
	  return true;
      }
    } else   {
      if (boolbv_get_width(expr.op0().type(), object_width))
	return true;
    }

    if (width > object_width && !int_encoding)
      bv = Z3_mk_zero_ext(z3_ctx, (width - object_width), pointer_object);
    else if (width < object_width && !int_encoding)
      bv = Z3_mk_extract(z3_ctx, (width - 1), 0, pointer_object);
    else
      bv = pointer_object;

    return false;
  } else if (object.id() == "index" && object.type().id() == "struct")     {
    const exprt &symbol = object.operands()[0];
    const exprt &index = object.operands()[1];
    Z3_ast array_value, array, array_index;

    if (convert_bv(symbol, array))
      return true;

    if (convert_bv(index, array_index))
      return true;

    array_value = Z3_mk_select(z3_ctx, array, array_index);
    pointer_object = z3_api.mk_tuple_select(z3_ctx, array_value, 0);

    const struct_typet &struct_type = to_struct_type(object.type());
    const struct_typet::componentst &components = struct_type.components();
    u_int i = 0;
    char val[2];

    for (struct_typet::componentst::const_iterator
         it = components.begin();
         it != components.end();
         it++, i++)
    {
      sprintf(val, "%i", i);
      if (int_encoding)
	pointer_object = z3_api.mk_int(z3_ctx, atoi(val));
      else
	pointer_object =
	  Z3_mk_int(z3_ctx, atoi(val),
	            Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));
    }
  } else   {
    if (convert_bv(object, pointer_object)) {
      if (object.type().id() == "pointer" && object.type().subtype().id() ==
          "symbol") {
	if (int_encoding)
	  bv = z3_api.mk_unsigned_int(z3_ctx, 0);
	else
	  bv =
	    Z3_mk_unsigned_int(z3_ctx, 0,
	                       Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));
	return false;
      } else
	return true;
    }

    if (object.type().id() == "signedbv" || object.type().id() == "unsignedbv"
        || object.type().id() == "fixedbv") {
      if (boolbv_get_width(object.type(), object_width))
	return true;

      if (width > object_width && !int_encoding)
	bv = Z3_mk_zero_ext(z3_ctx, (width - object_width), pointer_object);
      else if (width < object_width && !int_encoding)
	bv = Z3_mk_extract(z3_ctx, (width - 1), 0, pointer_object);
      else if (int_encoding && object.type().id() == "fixedbv")
	bv = Z3_mk_real2int(z3_ctx, pointer_object);
      else
	bv = pointer_object;

      return false;
    } else if (object.type().id() == "array")     {
      Z3_ast args[2];
      const exprt &object_array = expr.op0().operands()[0];
      const exprt &object_index = expr.op0().operands()[1];

      if (convert_bv(object_array, args[0]))
	return true;
      if (convert_bv(object_index, args[1]))
	return true;

      pointer_object = Z3_mk_select(z3_ctx, args[0], args[1]);

      if (object_array.type().subtype().id() == "pointer") {
	if (boolbv_get_width(object_array.type().subtype().subtype(),
	                     object_width))
	  return true;
      } else   {
	if (boolbv_get_width(object_array.type().subtype(), object_width))
	  return true;
      }

      if (width > object_width && !int_encoding)
	bv = Z3_mk_zero_ext(z3_ctx, (width - object_width), pointer_object);
      else if (width < object_width && !int_encoding)
	bv = Z3_mk_extract(z3_ctx, (width - 1), 0, pointer_object);
      else
	bv = pointer_object;

      return false;
    } else if (object.type().id() == "pointer")     {
      Z3_ast args[2];

      if (object.type().subtype().id() == "symbol") {
	bv = z3_api.mk_tuple_select(z3_ctx, pointer_object, 0);
	return false;
      }

      assert(object.operands().size() > 0);
      const exprt &object_array = object.operands()[0];

      if (object_array.type().id() == "array") {
	const exprt &object_index = object.operands()[1];

	if (convert_bv(object_array, args[0]))
	  return true;

	if (convert_bv(object_index, args[1]))
	  return true;

	pointer_object = Z3_mk_select(z3_ctx, args[0], args[1]);

	if (expr.op0().type().subtype().subtype().id() == "signedbv") {
	  if (boolbv_get_width(expr.op0().type().subtype().subtype(),
	                       object_width))
	    return true;
	} else if (expr.op0().type().subtype().id() == "empty" ||
	           expr.op0().type().subtype().id() == "pointer")       {
	  object_width = config.ansi_c.int_width;
	} else    {
	  if (boolbv_get_width(object.type().subtype(), object_width))
	    return true;
	}

	if (width > object_width && !int_encoding)
	  bv = Z3_mk_zero_ext(z3_ctx, (width - object_width), pointer_object);
	else if (width < object_width && !int_encoding)
	  bv = Z3_mk_extract(z3_ctx, (width - 1), 0, pointer_object);
	else /*if (width==object_width)*/
	  bv = pointer_object;

	return false;
      }
    } else if (object.type().id() == "struct" || object.type().id() ==
               "union")       {
      const struct_typet &struct_type = to_struct_type(object.type());
      const struct_typet::componentst &components = struct_type.components();

      pointer_object = z3_api.mk_tuple_select(z3_ctx, pointer_object, 0);

      if (object.type().id() == "union")
	pointer_object = z3_api.mk_tuple_select(z3_ctx, pointer_object,
	                                        components.size());

      for (struct_typet::componentst::const_iterator
           it = components.begin();
           it != components.end();
           it++)
      {
	if (it->type().id() == expr.type().id()) {
	  if (boolbv_get_width(it->type(), object_width))
	    return true;

	  if (width == object_width) {
	    if (convert_identifier(it->get("name").c_str(), it->type(), bv))
	      return true;

	    return false;
	  }
	} else   {
	  if (boolbv_get_width(it->type(), object_width))
	    return true;

	  if (width == object_width) {
	    if (convert_identifier(it->get("name").c_str(), expr.type(), bv /*pointer_object*/))
	      return true;

	    return false;
	  } else if (it->type().id() == "pointer")   {
	    if (convert_identifier(it->get("name").c_str(),
	                           /*expr.type()*/ it->type(), bv))
	      return true;
	    pointer_object = z3_api.mk_tuple_select(z3_ctx, bv, 0);
	  }
	}
      }

      if (width > object_width && !int_encoding)
	bv = Z3_mk_zero_ext(z3_ctx, (width - object_width), pointer_object);
      else if (width < object_width && !int_encoding)
	bv = Z3_mk_extract(z3_ctx, (width - 1), 0, pointer_object);
      else
	bv = pointer_object;

      return false;
    }
  }

  bv = pointer_object;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_zero_string_length

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_zero_string_length(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand;

  if (convert_bv(expr.op0(), operand))
    return true;

  bv = z3_api.mk_tuple_select(z3_ctx, operand, 0);

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_dynamic_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_is_dynamic_object(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand0, operand1;

  std::vector<unsigned> dynamic_objects;
  pointer_logic.get_dynamic_objects(dynamic_objects);

  if (dynamic_objects.empty()) {
    bv = Z3_mk_false(z3_ctx);
    return false;
  } else   {
    if (convert_bv(expr.op0(), operand0))
      return true;

    operand0 =
      Z3_mk_extract(z3_ctx, config.ansi_c.pointer_width + BV_ADDR_BITS - 1,
                    config.ansi_c.pointer_width,
                    operand0);

    if (dynamic_objects.size() == 1) {
      operand1 = convert_number(dynamic_objects.front(), BV_ADDR_BITS, true);
      bv = Z3_mk_eq(z3_ctx, operand0, operand1);
    } else   {
      unsigned i = 0, size;
      size = dynamic_objects.size() + 1;
      Z3_ast args[size];

      for (std::vector<unsigned>::const_iterator
           it = dynamic_objects.begin();
           it != dynamic_objects.end();
           it++, i++)
	args[i] = convert_number(*it, BV_ADDR_BITS, true);

      bv = Z3_mk_eq(z3_ctx, Z3_mk_or(z3_ctx, i, args), operand0);
    }
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_byte_update

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_byte_update(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 3);

  mp_integer i;
  if (to_integer(expr.op1(), i)) {
    if (convert_bv(expr.op0(), bv))
      return true;
    else
      return false;
    //throw "byte_update takes constant as second parameter";
  }

  Z3_ast tuple, value;
  uint width_op0, width_op2;

  if (convert_bv(expr.op0(), tuple))
    return true;

  if (convert_bv(expr.op2(), value))
    return true;

  if (boolbv_get_width(expr.op2().type(), width_op2))
    return true;

  std::stringstream s;
  s << i;

  DEBUGLOC;

  if (expr.op0().type().id() == "struct") {
    const struct_typet &struct_type = to_struct_type(expr.op0().type());
    const struct_typet::componentst &components = struct_type.components();
    bool has_field = false;

    assert(components.size() >= expr.op0().operands().size());
    assert(!components.empty());

    for (struct_typet::componentst::const_iterator
         it = components.begin();
         it != components.end();
         it++)
    {
      if (boolbv_get_width(it->type(), width_op0))
	return true;

      if ((it->type().id() == expr.op2().type().id()) &&
          (width_op0 == width_op2))
	has_field = true;
    }

    if (has_field)
      bv = z3_api.mk_tuple_update(z3_ctx, tuple, atoi(s.str().c_str()), value);
    else
      bv = tuple;
  } else if (expr.op0().type().id() == "signedbv")     {
    if (int_encoding) {
      bv = value;
      return false;
    }

    if (boolbv_get_width(expr.op0().type(), width_op0))
      return true;

    if (width_op0 == 0)
      throw "failed to get width of byte_update operand";

    if (width_op0 > width_op2)
      bv = Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op2), value);
    else
      throw "convert_byte_update: " + expr.id_string() + " is unsupported";
  } else
    throw "convert_byte_update: " + expr.id_string() + " is unsupported";

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_byte_extract

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_byte_extract(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);

  mp_integer i;
  if (to_integer(expr.op1(), i))
    return true;

  unsigned width, w;

  if (boolbv_get_width(expr.op0().type(), width))
    return true;

  if (boolbv_get_width(expr.type(), w))
    return true;

  if (width == 0)
    throw "failed to get width of byte_extract operand";

  mp_integer upper, lower;

  if (expr.id() == "byte_extract_little_endian") {
    upper = ((i + 1) * 8) - 1; //((i+1)*w)-1;
    lower = i * 8; //i*w;
  } else   {
    mp_integer max = width - 1;
    upper = max - (i * 8); //max-(i*w);
    lower = max - ((i + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  Z3_ast op0;

  if (convert_bv(expr.op0(), op0))
    return true;

  std::stringstream s_upper, s_lower, s_i;
  s_upper << upper;
  s_lower << lower;
  s_i << i;

  if (int_encoding) {
    if (expr.op0().type().id() == "fixedbv") {
      if (expr.type().id() == "signedbv" ||
          expr.type().id() == "unsignedbv") {
	Z3_ast tmp;
	op0 = Z3_mk_real2int(z3_ctx, op0);
	tmp = Z3_mk_int2bv(z3_ctx, width, op0);
	bv =
	  Z3_mk_extract(z3_ctx, atoi(s_upper.str().c_str()),
	                atoi(s_lower.str().c_str()), tmp);
	if (expr.type().id() == "signedbv")
	  bv = Z3_mk_bv2int(z3_ctx, bv, 1);
	else
	  bv = Z3_mk_bv2int(z3_ctx, bv, 0);
      } else
	throw "convert_byte_extract: " + expr.type().id_string() +
	      " is unsupported";
    } else if (expr.op0().type().id() == "signedbv" ||
               expr.op0().type().id() == "unsignedbv") {
      Z3_ast tmp;
      tmp = Z3_mk_int2bv(z3_ctx, width, op0);

      if (width >= atoi(s_upper.str().c_str()))
	bv =
	  Z3_mk_extract(z3_ctx, atoi(s_upper.str().c_str()),
	                atoi(s_lower.str().c_str()), tmp);
      else
	bv =
	  Z3_mk_extract(z3_ctx, atoi(s_upper.str().c_str()) -
	                atoi(s_lower.str().c_str()), 0, tmp);

      if (expr.op0().type().id() == "signedbv")
	bv = Z3_mk_bv2int(z3_ctx, bv, 1);
      else
	bv = Z3_mk_bv2int(z3_ctx, bv, 0);
    } else
      throw "convert_byte_extract: " + expr.op0().type().id_string() +
            " is unsupported";
  } else   {
    if (expr.op0().type().id() == "struct") {
      const struct_typet &struct_type = to_struct_type(expr.op0().type());
      const struct_typet::componentst &components = struct_type.components();
      unsigned i = 0;
      Z3_ast struct_elem[components.size() + 1],
             struct_elem_inv[components.size() + 1];

      for (struct_typet::componentst::const_iterator
           it = components.begin();
           it != components.end();
           it++, i++)
      {
	if (convert_bv(expr.op0().operands()[i], struct_elem[i]))
	  return true;
      }

      for (unsigned k = 0; k < components.size(); k++)
	struct_elem_inv[(components.size() - 1) - k] = struct_elem[k];

      for (unsigned k = 0; k < components.size(); k++)
      {
	if (k == 1)
	  struct_elem_inv[components.size()] = Z3_mk_concat(
	    z3_ctx, struct_elem_inv[k - 1], struct_elem_inv[k]);
	else if (k > 1)
	  struct_elem_inv[components.size()] = Z3_mk_concat(
	    z3_ctx, struct_elem_inv[components.size()], struct_elem_inv[k]);
      }
      op0 = struct_elem_inv[components.size()];
    }

    bv =
      Z3_mk_extract(z3_ctx, atoi(s_upper.str().c_str()), atoi(
                      s_lower.str().c_str()), op0);

    if (expr.op0().id() == "index") {
      Z3_ast args[2];

      const exprt &symbol = expr.op0().operands()[0];
      const exprt &index = expr.op0().operands()[1];

      if (convert_bv(symbol, args[0]))
	return true;

      if (convert_bv(index, args[1]))
	return true;

      bv = Z3_mk_select(z3_ctx, args[0], args[1]);

      unsigned width_expr;
      if (boolbv_get_width(expr.type(), width_expr))
	return true;

      if (width_expr > width) {
	if (expr.type().id() == "unsignedbv") {
	  bv = Z3_mk_zero_ext(z3_ctx, (width_expr - width), bv);
	}
      }

    }
  }

  DEBUGLOC;

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_isnan

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_isnan(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);

  const typet &op_type = expr.op0().type();

  if (op_type.id() == "fixedbv") {
    Z3_ast op0;
    unsigned width;

    if (boolbv_get_width(op_type, width))
      return true;

    if (convert_bv(expr.op0(), op0))
      return true;

    if (int_encoding)
      bv =
        Z3_mk_ite(z3_ctx,
                  Z3_mk_ge(z3_ctx,
                           Z3_mk_real2int(z3_ctx,
                                          op0), convert_number(0, width, true)),
                  Z3_mk_true(z3_ctx), Z3_mk_false(z3_ctx));
    else
      bv =
        Z3_mk_ite(z3_ctx, Z3_mk_bvsge(z3_ctx, op0, convert_number(0, width,
                                                                  true)),
                  Z3_mk_true(z3_ctx), Z3_mk_false(z3_ctx));
  } else
    throw "isnan with unsupported operand type";

  return false;
}

/*******************************************************************
   Function: z3_convt::convert_z3_expr

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::convert_z3_expr(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  if (expr.id() == "symbol")
    return convert_identifier(expr.get_string("identifier"), expr.type(), bv);
  else if (expr.id() == "nondet_symbol")
    return convert_identifier("nondet$" + expr.get_string(
                                "identifier"), expr.type(), bv);
  else if (expr.id() == "typecast")
    return convert_typecast(expr, bv);
  else if (expr.id() == "struct")
    return convert_struct(expr, bv);
  else if (expr.id() == "union")
    return convert_union(expr, bv);
  else if (expr.id() == "constant")
    return convert_constant(expr, bv);
  else if (expr.id() == "bitand" || expr.id() == "bitor" || expr.id() ==
           "bitxor"
           || expr.id() == "bitnand" || expr.id() == "bitnor" || expr.id() ==
           "bitnxor")
    return convert_bitwise(expr, bv);
  else if (expr.id() == "bitnot")
    return convert_bitnot(expr, bv);
  else if (expr.id() == "unary-")
    return convert_unary_minus(expr, bv);
  else if (expr.id() == "if")
    return convert_if(expr, bv);
  else if (expr.id() == "and" || expr.id() == "or" || expr.id() == "xor")
    return convert_logical_ops(expr, bv);
  else if (expr.id() == "not")
    return convert_logical_not(expr, bv);
  else if (expr.id() == "=" || expr.id() == "notequal")
    return convert_equality(expr, bv);
  else if (expr.id() == "<=" || expr.id() == "<" || expr.id() == ">="
           || expr.id() == ">")
    return convert_rel(expr, bv);
  else if (expr.id() == "+")
    return convert_add(expr, bv);
  else if (expr.id() == "-")
    return convert_sub(expr, bv);
  else if (expr.id() == "/")
    return convert_div(expr, bv);
  else if (expr.id() == "mod")
    return convert_mod(expr, bv);
  else if (expr.id() == "*")
    return convert_mul(expr, bv);
  else if (expr.id() == "address_of" || expr.id() == "implicit_address_of"
           || expr.id() == "reference_to")
    return convert_pointer(expr, bv);
  else if (expr.id() == "array_of")
    return convert_array_of(expr, bv);
  else if (expr.id() == "index")
    return convert_index(expr, bv);
  else if (expr.id() == "ashr" || expr.id() == "lshr" || expr.id() == "shl")
    return convert_shift(expr, bv);
  else if (expr.id() == "abs")
    return convert_abs(expr, bv);
  else if (expr.id() == "with")
    return convert_with(expr, bv);
  else if (expr.id() == "member")
    return convert_member(expr, bv);
  else if (expr.id() == "zero_string")
    return convert_zero_string(expr, bv);
  else if (expr.id() == "pointer_offset")
    return select_pointer_offset(expr, bv);
  else if (expr.id() == "pointer_object")
    return convert_pointer_object(expr, bv);
  else if (expr.id() == "same-object")
    return convert_object(expr, bv);
  else if (expr.id() == "string-constant") {
    exprt tmp;
    string2array(expr, tmp);
    return convert_bv(tmp, bv);
  } else if (expr.id() == "zero_string_length")
    return convert_zero_string_length(expr.op0(), bv);
  else if (expr.id() == "replication")
    assert(expr.operands().size() == 2);
  else if (expr.id() == "is_dynamic_object")
    return convert_is_dynamic_object(expr, bv);
  else if (expr.id() == "byte_update_little_endian" ||
           expr.id() == "byte_update_big_endian")
    return convert_byte_update(expr, bv);
  else if (expr.id() == "byte_extract_little_endian" ||
           expr.id() == "byte_extract_big_endian")
    return convert_byte_extract(expr, bv);
#if 1
  else if (expr.id() == "isnan")
    return convert_isnan(expr, bv);
#endif

  return true;
}

/*******************************************************************
   Function: z3_convt::assign_z3_expr

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

bool
z3_convt::assign_z3_expr(const exprt expr)
{
  u_int size = expr.operands().size();

  //ignore these IRep expressions for now. I don't know what they mean.

  if (size == 2 && expr.op1().id() == "unary+") {
    ignoring(expr.op1());
    ignoring_expr = false;
    return false;
  }

  return true;
}


/*******************************************************************
   Function: z3_convt::set_to

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::set_to(const exprt &expr, bool value)
{
  DEBUGLOC;

  //std::cout << "expr.pretty(): " << expr.pretty() << std::endl;

#if 1
  if (expr.type().id() != "bool") {
    std::string msg = "prop_convt::set_to got "
                      "non-boolean expression:\n";
    msg += expr.to_string();
    throw msg;
  }

  bool boolean = true;

  forall_operands(it, expr)
  if (it->type().id() != "bool") {
    boolean = false;
    break;
  }

  if (boolean) {
    if (expr.id() == "not") {
      if (expr.operands().size() == 1) {
	set_to(expr.op0(), !value);
	return;
      }
    } else   {
      if (value) {
	// set_to_true
	if (expr.id() == "and") {
	  forall_operands(it, expr)
	  set_to_true(*it);

	  return;
	} else if (expr.id() == "or")      {
	  if (expr.operands().size() > 0) {
	    bvt bv;
	    bv.reserve(expr.operands().size());

	    forall_operands(it, expr)
	    bv.push_back(convert(*it));
	    prop.lcnf(bv);
	    return;
	  }
	} else if (expr.id() == "=>")      {
	  if (expr.operands().size() == 2) {
	    bvt bv;
	    bv.resize(2);
	    bv[0] = prop.lnot(convert(expr.op0()));
	    bv[1] = convert(expr.op1());
	    prop.lcnf(bv);
	    return;
	  }
	}
      } else   {
	// set_to_false
	if (expr.id() == "=>") { // !(a=>b)  ==  (a && !b)
	  if (expr.operands().size() == 2) {
	    set_to_true(expr.op0());
	    set_to_false(expr.op1());
	  }
	} else if (expr.id() == "or")      { // !(a || b)  ==  (!a && !b)
	  forall_operands(it, expr)
	  set_to_false(*it);
	}
      }
    }
  }

  // fall back to convert
  if (expr.op1().id() != "with")
    prop.l_set_to(convert(expr), value);

#endif

  if (value && expr.id() == "and") {
    forall_operands(it, expr)
    set_to(*it, true);
    return;
  }

  if (value && expr.is_true())
    return;



  if (expr.id() == "=" && value) {
    assert(expr.operands().size() == 2);

    Z3_ast result, operand[2];
    const exprt &op0 = expr.op0();
    const exprt &op1 = expr.op1();

    if (assign_z3_expr(expr) && ignoring_expr) {
      if (convert_bv(op0, operand[0])) {
	ignoring(op0);
	return;
      }
      if (convert_bv(op1, operand[1])) {
	ignoring(op1);
	return;
      }

      if (op0.type().id() == "pointer" && op1.type().id() == "pointer") {
    	  Z3_ast pointer[2], formula[2];

    	  pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 0);         //select
    	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 // object
    	  pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 0);

    	  formula[0] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);

    	  pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);         //select
				                                            // index
    	  pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);
    	  formula[1] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);

    	  if (expr.op0().type().id() == "bool")
    		  result = Z3_mk_iff(z3_ctx, formula[0], formula[1]);
    	  else
    		  result = Z3_mk_and(z3_ctx, 2, formula);

    	  Z3_assert_cnstr(z3_ctx, result);

    	  if (z3_prop.smtlib)
    		  z3_prop.assumpt.push_back(result);
      } else   {
#if 1
	if (op0.type().id() == "union" && op1.id() == "with") {
	  union_vars.insert(std::pair<std::string,
	                              unsigned int>(op0.get_string("identifier"),
	                                            convert_member_name(
	                                              op1.op0(), op1.op1())));
	}
#endif

	if (op0.type().id() == "bool")
	  result = Z3_mk_iff(z3_ctx, operand[0], operand[1]);
	else
	  result = Z3_mk_eq(z3_ctx, operand[0], operand[1]);

	Z3_assert_cnstr(z3_ctx, result);

	if (z3_prop.smtlib)
	  z3_prop.assumpt.push_back(result);

	if (uw && expr.op0().get_string("identifier").find("guard_exec") !=
	    std::string::npos
	    && number_of_assumptions < max_core_size) {
	  if (!op1.is_true())
	    generate_assumptions(expr, operand[0]);
	}
      }
    }
  }
  ignoring_expr = true;

  DEBUGLOC;

}

/*******************************************************************
   Function: z3_convt::get_number_vcs_z3

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

u_int
z3_convt::get_number_vcs_z3(void)
{
  return number_vcs_z3;
}

/*******************************************************************
   Function: z3_convt::get_number_variables_z

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

u_int
z3_convt::get_number_variables_z3(void)
{
  return number_variables_z3;
}

Z3_context z3_convt::z3_ctx = NULL;

unsigned int z3_convt::num_ctx_ileaves = 0;
bool z3_convt::s_relevancy = false;
bool z3_convt::s_is_uw = false;
