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
#include <simplify_expr.h>
#include <solvers/flattening/boolbv_width.h>
#include <fixedbv.h>
#include <solvers/flattening/boolbv.h>
#include <solvers/flattening/boolbv_type.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

//static Z3_ast core[Z3_UNSAT_CORE_LIMIT];
static std::vector<Z3_ast> core_vector;
static u_int unsat_core_size = 0;
static u_int assumptions_status = 0;

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

    std::list<Z3_ast>::const_iterator it;
    unsigned int i;
    for (it = z3_prop.assumpt.begin(), i = 0; it != z3_prop.assumpt.end(); it++, i++) {
      assumpt_array[i] = *it;
    }

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

  if (!z3_prop.uw)
    Z3_pop(z3_ctx, 1);

  // Experimental: if we're handled say, 10,000 ileaves, refresh the z3 ctx.
  num_ctx_ileaves++;

  if (num_ctx_ileaves == 10000) {
    num_ctx_ileaves = 0;
    Z3_del_context(z3_ctx);
    finalize_symbols();
    Z3_reset_memory();
    z3_ctx = z3_api.mk_proof_context(s_is_uw);
  }
}

void
z3_convt::init_addr_space_array(void)
{
  Z3_symbol mk_tuple_name, proj_names[2];
  Z3_type_ast proj_types[2];
  Z3_const_decl_ast mk_tuple_decl, proj_decls[2];
  Z3_sort native_int_sort;

  addr_space_sym_num = 1;

  if (int_encoding) {
    native_int_sort = Z3_mk_int_type(z3_ctx);
  } else {
    native_int_sort = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  }

  proj_types[0] = proj_types[1] = native_int_sort;

  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, "struct_type_addr_space_tuple");
  proj_names[0] = Z3_mk_string_symbol(z3_ctx, "start");
  proj_names[1] = Z3_mk_string_symbol(z3_ctx, "end");

  addr_space_tuple_sort = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, 2,
                                           proj_names, proj_types,
                                           &mk_tuple_decl, proj_decls);

  // Generate initial array with all zeros for all fields.
  addr_space_arr_sort = Z3_mk_array_type(z3_ctx, native_int_sort,
                                         addr_space_tuple_sort);

  Z3_ast num = convert_number(0, config.ansi_c.int_width, true);
  Z3_ast initial_val = z3_api.mk_tuple(z3_ctx, addr_space_tuple_sort, num, num,
                                       NULL);

  Z3_ast initial_const = Z3_mk_const_array(z3_ctx, native_int_sort, initial_val);
  Z3_ast first_name = z3_api.mk_var(z3_ctx, "__ESBMC_addrspace_arr_0",
                                    addr_space_arr_sort);
  Z3_ast eq = Z3_mk_eq(z3_ctx, first_name, initial_const);
  assert_formula(eq);

  // Actually store into array
  Z3_ast obj_idx = convert_number(pointer_logic.get_null_object(),
                                  config.ansi_c.int_width, true);

  Z3_ast range_tuple = z3_api.mk_var(z3_ctx, "__ESBMC_ptr_addr_range_0",
                                     addr_space_tuple_sort);
  initial_val = z3_api.mk_tuple(z3_ctx, addr_space_tuple_sort, num, num, NULL);
  eq = Z3_mk_eq(z3_ctx, initial_val, range_tuple);
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.get_null_object(), range_tuple);

  // We also have to initialize the invalid object... however, I've no idea
  // what it /means/ yet, so go for some arbitary value.
  num = convert_number(1, config.ansi_c.int_width, true);
  range_tuple = z3_api.mk_var(z3_ctx, "__ESBMC_ptr_addr_range_1",
                              addr_space_tuple_sort);
  initial_val = z3_api.mk_tuple(z3_ctx, addr_space_tuple_sort, num, num, NULL);
  eq = Z3_mk_eq(z3_ctx, initial_val, range_tuple);
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.get_invalid_object(), range_tuple);

  // Associate the symbol "0" with the null object; this is necessary because
  // of the situation where 0 is valid as a representation of null, but the
  // frontend (for whatever reasons) converts it to a symbol rather than the
  // way it handles NULL (constant with val "NULL")
  Z3_sort pointer_type;
  create_pointer_type(pointer_type);
  Z3_ast zero_sym = z3_api.mk_var(z3_ctx, "0", pointer_type);
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, pointer_type);

  Z3_ast args[2];
  args[0] = Z3_mk_int(z3_ctx, 0, native_int_sort);
  args[1] = Z3_mk_int(z3_ctx, 0, native_int_sort);
  Z3_ast ptr_val = Z3_mk_app(z3_ctx, decl, 2, args);
  Z3_ast constraint = Z3_mk_eq(z3_ctx, zero_sym, ptr_val);
  assert_formula(constraint);

  // Record the fact that we've registered these objects
  addr_space_data[0] = 0;
  addr_space_data[1] = 0;

  return;
}

void
z3_convt::bump_addrspace_array(unsigned int idx, Z3_ast val)
{
  std::string str, new_str;

  str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num++);
  Z3_ast addr_sym = z3_api.mk_var(z3_ctx, str.c_str(), addr_space_arr_sort);
  Z3_ast obj_idx = convert_number(idx, config.ansi_c.int_width, true);

  Z3_ast store = Z3_mk_store(z3_ctx, addr_sym, obj_idx, val);

  new_str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num);
  Z3_ast new_addr_sym = z3_api.mk_var(z3_ctx, new_str.c_str(),
                                      addr_space_arr_sort);

  Z3_ast eq = Z3_mk_eq(z3_ctx, new_addr_sym, store);
  assert_formula(eq);

  return;
}

std::string
z3_convt::get_cur_addrspace_ident(void)
{

  std::string str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num);
  return str;
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
    z3_prop.assumpt.push_back(Z3_mk_not(z3_ctx, result));
  } else
    z3_prop.assumpt.push_back(Z3_mk_not(z3_ctx, result));

  // Ensure addrspace array makes its way to the output
  std::string sym = get_cur_addrspace_ident();
  Z3_ast addr_sym = z3_api.mk_var(z3_ctx, sym.c_str(), addr_space_arr_sort);
  z3_prop.assumpt.push_back(addr_sym);
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

void
z3_convt::finalize_pointer_chain(void)
{

  unsigned int offs, num_ptrs = addr_space_data.size();
  if (num_ptrs == 0)
    return;

  Z3_ast ptr_idxs[num_ptrs];

  Z3_sort native_int_sort;
  if (int_encoding)
    native_int_sort = Z3_mk_int_sort(z3_ctx);
  else
    native_int_sort = Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  offs = 2;
  for (std::map<unsigned,unsigned>::const_iterator it = addr_space_data.begin();
       it != addr_space_data.end(); it++) {

    Z3_ast start = z3_api.mk_var(z3_ctx,
                           ("__ESBMC_ptr_obj_start_" + itos(it->first)).c_str(),
                           native_int_sort);
    Z3_ast start_num = convert_number(offs, config.ansi_c.int_width, true);
    Z3_ast eq = Z3_mk_eq(z3_ctx, start, start_num);
    assert_formula(eq);

    offs += it->second - 1;

    Z3_ast end = z3_api.mk_var(z3_ctx,
                           ("__ESBMC_ptr_obj_end_" + itos(it->first)).c_str(),
                           native_int_sort);
    Z3_ast end_num = convert_number(offs, config.ansi_c.int_width, true);
    eq = Z3_mk_eq(z3_ctx, end, end_num);
    assert_formula(eq);

    offs++;
  }

  return;
}

Z3_lbool
z3_convt::check2_z3_properties(void)
{
  DEBUGLOC

  Z3_model m = 0;
  Z3_lbool result;
  unsigned i;
  Z3_ast proof, core[z3_prop.assumpt.size()],
         assumptions_core[z3_prop.assumpt.size()];
  std::string literal;

  assumptions_status = z3_prop.assumpt.size();

  if (z3_prop.uw) {
    std::list<Z3_ast>::const_iterator it;
    for (it = z3_prop.assumpt.begin(), i = 0; it != z3_prop.assumpt.end(); it++, i++) {
      assumptions_core[i] = *it;
    }
  }

  try
  {
    if (z3_prop.uw) {
      unsat_core_size = z3_prop.assumpt.size();
      memset(core, 0, sizeof(Z3_ast) * unsat_core_size);
      result = Z3_check_assumptions(z3_ctx, z3_prop.assumpt.size(),
                             assumptions_core, &m, &proof, &unsat_core_size,
                             core);
    } else {
      result = Z3_check_and_get_model(z3_ctx, &m);
    }
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
    abort();
  }


  if (result == Z3_L_TRUE) {
    store_sat_assignments(m);
    //printf("Counterexample:\n");
    //z3_api.display_model(z3_ctx, stdout, m);
  } else if (z3_prop.uw && result == Z3_L_FALSE)   {
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

  return result;
}

void
z3_convt::link_syms_to_literals(void)
{

  symbolst::const_iterator it;
  for (it = symbols.begin(); it != symbols.end(); it++) {
    // Generate an equivalence between the symbol and the literal
    Z3_ast sym = z3_api.mk_var(z3_ctx, it->first.as_string().c_str(),
                                Z3_mk_bool_sort(z3_ctx));
    Z3_ast formula = Z3_mk_iff(z3_ctx, z3_prop.z3_literal(it->second), sym);
    assert_formula(formula);
 }
}

void
z3_convt::assert_formula(Z3_ast ast, bool needs_literal)
{

  z3_prop.assert_formula(ast, needs_literal);
  return;
}

void
z3_convt::assert_literal(literalt l, Z3_ast formula)
{

  z3_prop.assert_literal(l, formula);
  return;
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

void
z3_convt::select_pointer_value(Z3_ast object, Z3_ast offset, Z3_ast &bv)
{
  DEBUGLOC;

  bv = Z3_mk_select(z3_ctx, object, offset);
  return;
}

/*******************************************************************
   Function: z3_convt::create_z3_array_var

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::create_array_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  Z3_sort elem_sort, idx_sort;

  if (int_encoding) {
    idx_sort = Z3_mk_int_type(z3_ctx);
  } else {
    idx_sort = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  }

  create_type(type.subtype(), elem_sort);
  bv = Z3_mk_array_type(z3_ctx, idx_sort, elem_sort);
  DEBUGLOC;

  return;
}

/*******************************************************************
   Function: z3_convt::create_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::create_type(const typet &type, Z3_type_ast &bv)
{
  DEBUGLOC;

  unsigned width = config.ansi_c.int_width;

  if (type.id() == "bool") {
    bv = Z3_mk_bool_type(z3_ctx);
  } else if (type.id() == "signedbv" || type.id() == "unsignedbv" ||
             type.id() == "c_enum" || type.id() == "incomplete_c_enum") {
    get_type_width(type, width);

    if (int_encoding)
      bv = Z3_mk_int_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, width);
  } else if (type.id() == "fixedbv")   {
    get_type_width(type, width);

    if (int_encoding)
      bv = Z3_mk_real_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, width);
  } else if (type.id() == "array")     {
    create_array_type(type, bv);
  } else if (type.id() == "struct")     {
    create_struct_type(type, bv);
  } else if (type.id() == "union")     {
    create_union_type(type, bv);
  } else if (type.id() == "pointer")     {
    create_pointer_type(bv);
  } else if (type.id() == "symbol" || type.id() == "empty" ||
             type.id() == "c_enum")     {
    if (int_encoding)
      bv = Z3_mk_int_type(z3_ctx);
    else
      bv = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  } else
    throw new conv_error("unexpected type in create_type", type);

  DEBUGLOC;

  return;
}

/*******************************************************************
   Function: z3_convt::create_struct_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::create_struct_union_type(const typet &type, bool uni, Z3_type_ast &bv)
{
  DEBUGLOC;

  Z3_symbol mk_tuple_name, *proj_names;
  std::string name;
  Z3_type_ast *proj_types;
  Z3_const_decl_ast mk_tuple_decl, *proj_decls;
  u_int num_elems;

  const struct_union_typet &su_type = to_struct_union_type(type);
  const struct_union_typet::componentst &components = su_type.components();

  assert(components.size() > 0);
  num_elems = components.size();
  if (uni) num_elems++;

  proj_names = new Z3_symbol[num_elems];
  proj_types = new Z3_type_ast[num_elems];
  proj_decls = new Z3_const_decl_ast[num_elems];

  name = ((uni) ? "union" : "struct" );
  name += "_type_" + type.get_string("tag");
  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, name.c_str());

  u_int i = 0;
  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    proj_names[i] = Z3_mk_string_symbol(z3_ctx, it->get("name").c_str());
    create_type(it->type(), proj_types[i]);
  }

  if (uni) {
    // ID field records last value written to union
    proj_names[num_elems - 1] = Z3_mk_string_symbol(z3_ctx, "id");
    // XXXjmorse - must this field really become a bitfield, ever? It's internal
    // tracking data, not program data.
    if (int_encoding)
      proj_types[num_elems - 1] = Z3_mk_int_type(z3_ctx);
    else
      proj_types[num_elems - 1] = Z3_mk_bv_type(z3_ctx,
                                                config.ansi_c.int_width);
  }

  bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, num_elems, proj_names,
                        proj_types, &mk_tuple_decl, proj_decls);

  delete[] proj_names;
  delete[] proj_types;
  delete[] proj_decls;

  DEBUGLOC;

  return;
}
/*******************************************************************
   Function: z3_convt::create_pointer_type

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::create_pointer_type(Z3_type_ast &bv)
{
  DEBUGLOC;

  // XXXjmorse - this assertion should never fail, but it does...
  //assert(type.id() == "pointer" || type.id() == "reference");

  Z3_symbol mk_tuple_name, proj_names[2];
  Z3_type_ast proj_types[2];
  Z3_const_decl_ast mk_tuple_decl, proj_decls[2];
  Z3_sort native_int_sort;

  if (int_encoding) {
    native_int_sort = Z3_mk_int_type(z3_ctx);
  } else {
    native_int_sort = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  }

  proj_types[0] = proj_types[1] = native_int_sort;

  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, "pointer_tuple");
  proj_names[0] = Z3_mk_string_symbol(z3_ctx, "object");
  proj_names[1] = Z3_mk_string_symbol(z3_ctx, "index");

  bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, 2, proj_names, proj_types,
                        &mk_tuple_decl, proj_decls);

  DEBUGLOC;

  return;
}

/*******************************************************************
   Function: z3_convt::convert_identifier

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_identifier(const std::string &identifier, const typet &type,
  Z3_ast &bv)
{
  DEBUGLOC;

  Z3_sort sort;

  // References to unsigned int identifiers need to be assumed to be > 0,
  // otherwise the solver is free to assign negative nums to it.
  if (type.id() == "unsignedbv" && int_encoding) {
    Z3_ast formula;
    bv = z3_api.mk_int_var(z3_ctx, identifier.c_str());
    formula = Z3_mk_ge(z3_ctx, bv, z3_api.mk_int(z3_ctx, 0));
    assert_formula(formula);
    return;
  }

  create_type(type, sort);
  bv = z3_api.mk_var(z3_ctx, identifier.c_str(), sort);

  DEBUGLOC;
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
    return true;
  }

  return false;
}

/*******************************************************************\

   Function: z3_convt::convert_bv Inputs:

   Outputs:

   Purpose:

\*******************************************************************/

void
z3_convt::convert_bv(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  bv_cachet::const_iterator cache_result = bv_cache.find(expr);
  if (cache_result != bv_cache.end()) {
    bv = cache_result->second;
    return;
  }

  convert_z3_expr(expr, bv);

  // insert into cache
  bv_cache.insert(std::pair<const exprt, Z3_ast>(expr, bv));

  DEBUGLOC;

  return;
}

/*******************************************************************
   Function: z3_convt::read_cache

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::read_cache(const exprt &expr, Z3_ast &bv)
{
  std::string symbol;

  symbol = expr.get_string("identifier");

  for (z3_cachet::const_iterator it = z3_cache.begin();
       it != z3_cache.end(); it++)
  {
    if (symbol.compare((*it).second.c_str()) == 0) {
      convert_bv((*it).first, bv);
      return;
    }
  }

  convert_bv(expr, bv);
}

/*******************************************************************
   Function: z3_convt::write_cache

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
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
      return;
    }
  }
}

/*******************************************************************
   Function: z3_convt::convert_cmp

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_cmp(const exprt &expr)
{
  DEBUGLOC;

  typedef Z3_ast (*calltype)(Z3_context ctx, Z3_ast op1, Z3_ast op2);
  calltype signedbvcall, unsignedbvcall, intcall;

  if (expr.id() == "<=") {
    intcall = Z3_mk_le;
    signedbvcall = Z3_mk_bvsle;
    unsignedbvcall = Z3_mk_bvule;
  } else if (expr.id() == ">=") {
    intcall = Z3_mk_ge;
    signedbvcall = Z3_mk_bvsge;
    unsignedbvcall = Z3_mk_bvuge;
  } else if (expr.id() == "<") {
    intcall = Z3_mk_lt;
    signedbvcall = Z3_mk_bvslt;
    unsignedbvcall = Z3_mk_bvult;
  } else if (expr.id() == ">") {
    intcall = Z3_mk_gt;
    signedbvcall = Z3_mk_bvsgt;
    unsignedbvcall = Z3_mk_bvugt;
  } else {
    throw new conv_error("Unexpected expr in convert_cmp", expr);
  }

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];

  convert_bv(expr.op0(), operand[0]);
  convert_bv(expr.op1(), operand[1]);

  // XXXjmorse - pointer comparison needs serious consideration
  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  if (int_encoding) {
    bv = intcall(z3_ctx, operand[0], operand[1]);
  } else   {
    if (expr.op1().type().id() == "signedbv" || expr.op1().type().id() ==
        "fixedbv" || expr.op1().type().id() == "pointer") {
      bv = signedbvcall(z3_ctx, operand[0], operand[1]);
    } else if (expr.op1().type().id() == "unsignedbv") {
      bv = unsignedbvcall(z3_ctx, operand[0], operand[1]);
    } else {
      throw new conv_error("Unexpected type in convert_cmp", expr);
    }
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

  assert(expr.operands().size() == 2);
  Z3_ast bv, operand[2];
  const exprt &op0 = expr.op0();
  const exprt &op1 = expr.op1();

  if (op0.type().id() == "array")
    write_cache(op0);

  convert_bv(op0, operand[0]);
  convert_bv(op1, operand[1]);

  if (expr.id() == "=")
    bv = Z3_mk_eq(z3_ctx, operand[0], operand[1]);
  else
    bv = Z3_mk_distinct(z3_ctx, 2, operand);

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

  Z3_ast bv, pointer[2], objs[2];
  const exprt &op0 = expr.op0();
  const exprt &op1 = expr.op1();

  convert_bv(op0, pointer[0]);
  convert_bv(op1, pointer[1]);
  objs[0] = z3_api.mk_tuple_select(z3_ctx, pointer[0], 0);
  objs[1] = z3_api.mk_tuple_select(z3_ctx, pointer[1], 0);
  bv = Z3_mk_eq(z3_ctx, objs[0], objs[1]);

  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_is_dynamic_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_is_dynamic_object(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);

  exprt sym("symbol", array_typet());
  sym.type().subtype() = bool_typet();
  sym.set("identifier", "__ESBMC_is_dynamic");
  exprt pointerobj("pointer_object", signedbv_typet());
  exprt ptrsrc = expr.op0();
  pointerobj.move_to_operands(ptrsrc);
  exprt index("index", bool_typet());
  index.move_to_operands(sym, pointerobj);

  Z3_ast bv;
  convert_bv(index, bv);
  return bv;
}

/*******************************************************************
   Function: z3_convt::convert_overflow_sum_sub_mul

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

Z3_ast
z3_convt::convert_overflow_sum_sub_mul(const exprt &expr)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast bv, result[2], operand[2];
  unsigned width_op0, width_op1;

  if (expr.op0().type().id() == "array")
    write_cache(expr.op0());

  convert_bv(expr.op0(), operand[0]);

  if (expr.op0().type().id() == "pointer")
    operand[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

  convert_bv(expr.op1(), operand[1]);

  if (expr.op1().type().id() == "pointer")
    operand[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);

  get_type_width(expr.op0().type(), width_op0);
  get_type_width(expr.op1().type(), width_op1);

  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width_op0, operand[0]);
    operand[1] = Z3_mk_int2bv(z3_ctx, width_op1, operand[1]);
  }

  typedef Z3_ast (*type1)(Z3_context, Z3_ast, Z3_ast, Z3_bool);
  typedef Z3_ast (*type2)(Z3_context, Z3_ast, Z3_ast);
  type1 call1;
  type2 call2;

  if (expr.id() == "overflow-+") {
    call1 = Z3_mk_bvadd_no_overflow;
    call2 = Z3_mk_bvadd_no_underflow;
  } else if (expr.id() == "overflow--") {
    call1 = Z3_mk_bvsub_no_underflow;
    call2 = Z3_mk_bvsub_no_overflow;
  } else if (expr.id() == "overflow-*") {
    call1 = Z3_mk_bvmul_no_overflow;
    call2 = Z3_mk_bvmul_no_underflow;
  } else {
    assert(false);
  }

  if (expr.op0().type().id() == "signedbv" && expr.op1().type().id() ==
      "signedbv")
    result[0] = call1(z3_ctx, operand[0], operand[1], 1);
  else if (expr.op0().type().id() == "unsignedbv" && expr.op1().type().id() ==
           "unsignedbv")
    result[0] = call1(z3_ctx, operand[0], operand[1], 0);

  result[1] = call2(z3_ctx, operand[0], operand[1]);
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

  convert_bv(expr.op0(), operand);

  if (expr.op0().type().id() == "pointer")
    operand = z3_api.mk_tuple_select(z3_ctx, operand, 1);

  get_type_width(expr.op0().type(), width);

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

  // Addition in the following strips the "overflow-typecast-" prefix, and
  // leaves us with the number of bits in the irep name.
  unsigned bits = atoi(expr.id().c_str() + 18);

  const exprt::operandst &operands = expr.operands();

  if (operands.size() != 1)
    throw "operand " + expr.id_string() + " takes one operand";

  Z3_ast bv, operand[3], mid, overflow[2], tmp, minus_one, two;
  uint64_t result;
  u_int width;

  get_type_width(expr.op0().type(), width);

  if (bits >= width || bits == 0)
    throw "overflow-typecast got wrong number of bits";

  assert(bits <= 32 && bits != 0);
  result = 1 << bits;

  convert_bv(expr.op0(), operand[0]);

  if (int_encoding)
    operand[0] = Z3_mk_int2bv(z3_ctx, width, operand[0]);

  // XXXjmorse - fixedbv is /not/ always partitioned at width/2
  if (expr.op0().type().id() == "fixedbv") {
    unsigned size = (width / 2) + 1;
    operand[0] = Z3_mk_extract(z3_ctx, width - 1, size - 1, operand[0]);
  }

  if (expr.op0().type().id() == "signedbv" || expr.op0().type().id() ==
      "fixedbv") {
    // Produce some useful constants
    unsigned int nums_width = (expr.op0().type().id() == "signedbv")
                               ? width : width / 2;
    tmp = convert_number_bv(result, nums_width, true);
    two = convert_number_bv(2, nums_width, true);
    minus_one = convert_number_bv(-1, nums_width, true);

    // Now produce numbers that bracket the selected bitwidth. So for 16 bis
    // we would generate 2^15-1 and -2^15
    mid = Z3_mk_bvsdiv(z3_ctx, tmp, two);
    operand[1] = Z3_mk_bvsub(z3_ctx, mid, minus_one);
    operand[2] = Z3_mk_bvmul(z3_ctx, operand[1], minus_one);

    // Ensure operand lies between these braces
    overflow[0] = Z3_mk_bvslt(z3_ctx, operand[0], operand[1]);
    overflow[1] = Z3_mk_bvsgt(z3_ctx, operand[0], operand[2]);
  } else if (expr.op0().type().id() == "unsignedbv")     {
    // Create zero and 2^bitwidth,
    operand[2] = convert_number_bv(0, width, false);
    operand[1] = convert_number_bv(result, width, false);
    // Ensure operand lies between those numbers.
    overflow[0] = Z3_mk_bvult(z3_ctx, operand[0], operand[1]);
    overflow[1] = Z3_mk_bvuge(z3_ctx, operand[0], operand[2]);
  }

  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, overflow));

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

  convert_bv(array, operand0);
  convert_bv(index, operand1);

  bv = Z3_mk_select(z3_ctx, operand0, operand1);

  return bv;
}

Z3_ast
z3_convt::convert_width(const exprt &expr)
{
  Z3_sort native_int_sort;
  unsigned int width;

  assert(expr.id() == "width");
  assert(expr.operands().size() == 1);

  get_type_width(expr.op0().type(), width);
  if (int_encoding)
    native_int_sort = Z3_mk_int_sort(z3_ctx);
  else
    native_int_sort = Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  return Z3_mk_int(z3_ctx, width, native_int_sort);
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

  literalt l = z3_prop.new_variable();
  Z3_ast formula, constraint;

  try {
    if (!assign_z3_expr(expr) && !ignoring_expr)
      return l;

    if (expr.id() == "is_zero_string") {
      ignoring(expr);
      return l;
    }

    convert_bv(expr, constraint);
  } catch (conv_error *e) {
    std::cerr << e->to_string() << std::endl;
    ignoring(expr);
    return l;
  }

  formula = Z3_mk_iff(z3_ctx, z3_prop.z3_literal(l), constraint);

  // While we have a literal, don't assert that it's true, only the link
  // between the formula and the literal. Otherwise, we risk asserting that a
  // formula within a assertion-statement is true or false.
  assert_formula(formula);
  return l;
}

/*******************************************************************
   Function: z3_convt::convert_typecast

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
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
  } else {
    throw new conv_error("Unimplemented bool typecast", expr);
  }
}


void
z3_convt::convert_typecast_fixedbv_nonint(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast args[2];

  const fixedbv_typet &fixedbv_type = to_fixedbv_type(expr.type());
  unsigned to_fraction_bits = fixedbv_type.get_fraction_bits();
  unsigned to_integer_bits = fixedbv_type.get_integer_bits();

  if (op.type().id() == "pointer") {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  if (op.type().id() == "unsignedbv" ||
      op.type().id() == "signedbv" ||
      op.type().id() == "enum") {
    unsigned from_width;

    get_type_width(op.type(), from_width);

    if (from_width == to_integer_bits) {
      convert_bv(op, bv);
    } else if (from_width > to_integer_bits)      {
      convert_bv(op, args[0]);
      bv = Z3_mk_extract(z3_ctx, (from_width - 1), to_integer_bits, args[0]);
    } else   {
      assert(from_width < to_integer_bits);
      convert_bv(op, args[0]);
      bv = Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_width), args[0]);
    }

    bv = Z3_mk_concat(z3_ctx, bv, convert_number(0, to_fraction_bits, true));
  } else if (op.type().id() == "bool")      {
    Z3_ast zero, one;
    unsigned width;

    get_type_width(expr.type(), width);

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
      convert_bv(op, args[0]);

      magnitude =
        Z3_mk_extract(z3_ctx, (from_fraction_bits + to_integer_bits - 1),
                      from_fraction_bits, args[0]);
    } else   {
      assert(to_integer_bits > from_integer_bits);

      convert_bv(op, args[0]);

      magnitude =
        Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_integer_bits),
                       Z3_mk_extract(z3_ctx, from_width - 1, from_fraction_bits,
                                     args[0]));
    }

    if (to_fraction_bits <= from_fraction_bits) {
      convert_bv(op, args[0]);

      fraction =
        Z3_mk_extract(z3_ctx, (from_fraction_bits - 1),
                      from_fraction_bits - to_fraction_bits,
                      args[0]);
    } else   {
      assert(to_fraction_bits > from_fraction_bits);
      convert_bv(op, args[0]);

      fraction =
        Z3_mk_concat(z3_ctx,
                     Z3_mk_extract(z3_ctx, (from_fraction_bits - 1), 0,args[0]),
                     convert_number(0, to_fraction_bits - from_fraction_bits,
                                    true));
    }
    bv = Z3_mk_concat(z3_ctx, magnitude, fraction);
  } else {
    throw new conv_error("unexpected typecast to fixedbv", expr);
  }

  return;
}

void
z3_convt::convert_typecast_to_ints(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast args[2];
  unsigned to_width;

  get_type_width(expr.type(), to_width);

  if (op.type().id() == "signedbv" || op.type().id() == "c_enum" ||
      op.type().id() == "fixedbv") {
    unsigned from_width;
    get_type_width(op.type(), from_width);

    if (from_width == to_width) {
      convert_bv(op, bv);

      if (int_encoding && op.type().id() == "signedbv" &&
               expr.type().id() == "fixedbv")
	bv = Z3_mk_int2real(z3_ctx, bv);
      else if (int_encoding && op.type().id() == "fixedbv" &&
               expr.type().id() == "signedbv")
	bv = Z3_mk_real2int(z3_ctx, bv);
      // XXXjmorse - there isn't a case here for if !int_encoding

    } else if (from_width < to_width)      {
      convert_bv(op, args[0]);

      if (int_encoding &&
          ((expr.type().id() == "fixedbv" && op.type().id() == "signedbv")))
	bv = Z3_mk_int2real(z3_ctx, args[0]);
      else if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_sign_ext(z3_ctx, (to_width - from_width), args[0]);
    } else if (from_width > to_width)     {
      convert_bv(op, args[0]);

      if (int_encoding &&
          ((op.type().id() == "signedbv" && expr.type().id() == "fixedbv")))
	bv = Z3_mk_int2real(z3_ctx, args[0]);
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
  } else if (op.type().id() == "unsignedbv") {
    unsigned from_width;

    get_type_width(op.type(), from_width);

    if (from_width == to_width) {
      convert_bv(op, bv);
    } else if (from_width < to_width)      {
      convert_bv(op, args[0]);

      if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_zero_ext(z3_ctx, (to_width - from_width), args[0]);
    } else if (from_width > to_width)     {
      convert_bv(op, args[0]);

      if (int_encoding)
	bv = args[0];
      else
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, args[0]);
    }
  } else if (op.type().id() == "bool")     {
    Z3_ast zero = 0, one = 0;
    unsigned width;

    get_type_width(expr.type(), width);

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
    } else if (expr.type().id() == "fixedbv") {
      zero = Z3_mk_numeral(z3_ctx, "0", Z3_mk_real_type(z3_ctx));
      one = Z3_mk_numeral(z3_ctx, "1", Z3_mk_real_type(z3_ctx));
    } else {
      throw new conv_error("Unexpected type in typecast of bool", expr);
    }
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
  } else   {
    throw new conv_error("Unexpected type in int/ptr typecast", expr);
  }
}

void
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
	get_type_width(it->type(), width);

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
          throw new conv_error("Unexpected type when casting struct", *it);
	}
	j++;
      }
    }
  }
  exprt new_struct("symbol", s);
  new_struct.type().set("tag", expr.type().get_string("tag"));
  new_struct.set("identifier", "typecast_" + expr.op0().get_string("identifier"));

  convert_bv(new_struct, operand);

  i2 = 0;
  for (struct_typet::componentst::const_iterator
       it2 = components2.begin();
       it2 != components2.end();
       it2++, i2++)
  {
    Z3_ast formula;
    formula = Z3_mk_eq(z3_ctx, z3_api.mk_tuple_select(z3_ctx, operand, i2),
                       z3_api.mk_tuple_select(z3_ctx, bv, i2));
    assert_formula(formula);
  }

  bv = operand;
  return;
}

void
z3_convt::convert_typecast_enum(const exprt &expr, Z3_ast &bv)
{
  const exprt &op = expr.op0();
  Z3_ast zero, one;
  unsigned width;

  if (op.type().id() == "bool") {
    get_type_width(expr.type(), width);

    zero = convert_number(0, width, true);
    one =  convert_number(1, width, true);
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
  }
}

void
z3_convt::convert_typecast_to_ptr(const exprt &expr, Z3_ast &bv)
{

  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if (expr.op0().type().id() == "pointer") {
    // Yup.
    convert_bv(expr.op0(), bv);
    return;
  }

  // Unpleasentness; we don't know what pointer this integer is going to
  // correspond to, and there's no way of telling statically, so we have
  // to enumerate all pointers it could point at. IE, all of them. Which
  // is expensive, but here we are.

  // First cast it to an unsignedbv
  Z3_ast target;
  unsignedbv_typet int_type(config.ansi_c.int_width);
  typecast_exprt cast(int_type);
  cast.op() = expr.op0();
  convert_bv(cast, target);

  // Construct array for all possible object outcomes
  Z3_ast is_in_range[addr_space_data.size()];
  Z3_ast obj_ids[addr_space_data.size()];
  Z3_ast obj_starts[addr_space_data.size()];

  // Get symbol for current array of addrspace data
  std::string arr_sym_name = get_cur_addrspace_ident();
  Z3_ast addr_sym = z3_api.mk_var(z3_ctx, arr_sym_name.c_str(),
                                  addr_space_arr_sort);

  Z3_sort native_int_sort;
  if (int_encoding) {
    native_int_sort = Z3_mk_int_type(z3_ctx);
  } else {
    native_int_sort = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  }

  std::map<unsigned,unsigned>::const_iterator it;
  unsigned int i;
  for (it = addr_space_data.begin(), i = 0;
       it != addr_space_data.end(); it++, i++)
  {
    Z3_ast args[2];

    unsigned id = it->first;
    Z3_ast idx = convert_number(id, config.ansi_c.int_width, true);
    obj_ids[i] = idx;
    Z3_ast start = z3_api.mk_var(z3_ctx,
                                 ("__ESBMC_ptr_obj_start_" + itos(id)).c_str(),
                                 native_int_sort);
    Z3_ast end = z3_api.mk_var(z3_ctx,
                                 ("__ESBMC_ptr_obj_end_" + itos(id)).c_str(),
                                 native_int_sort);
    obj_starts[i] = start;

    if (int_encoding) {
      args[0] = Z3_mk_ge(z3_ctx, target, start);
      args[1] = Z3_mk_le(z3_ctx, target, end);
    } else {
      args[0] = Z3_mk_bvuge(z3_ctx, target, start);
      args[1] = Z3_mk_bvule(z3_ctx, target, end);
    }

    is_in_range[i] = Z3_mk_and(z3_ctx, 2, args);
  }

  // Generate a big ITE chain, selecing a particular pointer offset. A
  // significant question is what happens when it's neither; in which case I
  // suggest the ptr becomes invalid_object. However, this needs frontend
  // support to check for invalid_object after all dereferences XXXjmorse.
  Z3_sort pointer_sort;
  create_pointer_type(pointer_sort);

  // So, what's the default value going to be if it doesn't match any existing
  // pointers? Answer, it's going to be the invalid object identifier, but with
  // an offset that calculates to the integer address of this object.
  // That's so that we can store an invalid pointer in a pointer type, that
  // eventually can be converted back via some mechanism to a valid pointer.
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, pointer_sort);
  Z3_ast args[2];
  args[0] = convert_number(pointer_logic.get_invalid_object(),
                           config.ansi_c.int_width, true);

  // Calculate ptr offset - target minus start of invalid range, ie 1
  Z3_ast subargs[2];
  subargs[0] = target;
  subargs[1] = convert_number(1, config.ansi_c.int_width, false);

  if (int_encoding) {
    args[1] = Z3_mk_sub(z3_ctx, 2, subargs);
  } else {
    args[1] = Z3_mk_bvsub(z3_ctx, subargs[0], subargs[1]);
  }

  Z3_ast prev_in_chain = Z3_mk_app(z3_ctx, decl, 2, args);

  // Now that big ite chain,
  for (i = 0; i < addr_space_data.size(); i++) {
    args[0] = obj_ids[i];

    // Calculate ptr offset were it this
    if (int_encoding) {
      Z3_ast tmp_args[2];
      tmp_args[0] = target;
      tmp_args[1] = obj_starts[i];
      args[1] = Z3_mk_sub(z3_ctx, 2, tmp_args);
    } else {
      args[1] = Z3_mk_bvsub(z3_ctx, target, obj_starts[i]);
    }

    Z3_ast selected_tuple = Z3_mk_app(z3_ctx, decl, 2, args);

    prev_in_chain =
      Z3_mk_ite(z3_ctx, is_in_range[i], selected_tuple, prev_in_chain);
  }

  // Finally, we're now at the point where prev_in_chain represents a pointer
  // object. Hurrah.
  bv = prev_in_chain;
}

void
z3_convt::convert_typecast_from_ptr(const exprt &expr, Z3_ast &bv)
{
  unsignedbv_typet int_type(config.ansi_c.int_width);

  // The plan: index the object id -> address-space array and pick out the
  // start address, then add it to any additional pointer offset.

  // Generate type of address space array
  struct_typet strct;
  struct_union_typet::componentt cmp;
  cmp.type() = int_type;
  cmp.set_name("start");
  strct.components().push_back(cmp);
  cmp.set_name("end");
  strct.components().push_back(cmp);
  strct.set("tag", "addr_space_tuple");

  array_typet arr;
  arr.subtype() = strct;

  exprt obj_num("pointer_object", int_type);
  obj_num.copy_to_operands(expr.op0());

  symbol_exprt sym_arr(get_cur_addrspace_ident(), arr);
  index_exprt idx(strct);
  idx.array() = sym_arr;
  idx.index() = obj_num;

  // We've now grabbed the pointer struct, now get first element
  member_exprt memb(int_type);
  memb.op0() = idx;
  memb.set_component_name("start");

  exprt ptr_offs("pointer_offset", int_type);
  ptr_offs.copy_to_operands(expr.op0());
  exprt add("+", int_type);
  add.copy_to_operands(memb, ptr_offs);

  // Finally, replace typecast
  typecast_exprt cast(expr.type());
  cast.op() = add;
  convert_bv(cast, bv);
}

void
z3_convt::convert_typecast(const exprt &expr, Z3_ast &bv)
{
  assert(expr.operands().size() == 1);
  const exprt &op = expr.op0();

  convert_bv(op, bv);

  if (expr.type().id() == "pointer") {
    convert_typecast_to_ptr(expr, bv);
  } else if (expr.op0().type().id() == "pointer") {
    convert_typecast_from_ptr(expr, bv);
  } else if (expr.type().id() == "bool") {
    convert_typecast_bool(expr, bv);
  } else if (expr.type().id() == "fixedbv" && !int_encoding)      {
    convert_typecast_fixedbv_nonint(expr, bv);
  } else if ((expr.type().id() == "signedbv" || expr.type().id() ==
              "unsignedbv"
              || expr.type().id() == "fixedbv" || expr.type().id() ==
              "pointer")) {
    convert_typecast_to_ints(expr, bv);
  } else if (expr.type().id() == "struct")     {
    convert_typecast_struct(expr, bv);
  } else if (expr.type().id() == "c_enum")      {
    convert_typecast_enum(expr, bv);
  } else {
    // XXXjmorse -- what about all other types, eh?
    throw new conv_error("Typecast for unexpected type", expr);
  }
}

/*******************************************************************
   Function: z3_convt::convert_struct_union

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_struct_union(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  Z3_ast value;

  // XXXjmorse - original convert_struct and convert_union were used both when
  // either expr.id() or expr.type().id() was struct/union. This can (still)
  // lead to an arbitary irep with type struct/union being fed here, which
  // could be invalid.

  // Converts a static struct/union - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  const struct_typet &struct_type = to_struct_type(expr.type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0;
  std::string identifier;

  assert(components.size() >= expr.operands().size());
  assert(!components.empty());

  if (expr.id() == "struct" || expr.type().id() == "struct")
    identifier = "conv_struct_" + expr.type().get_string("tag");
  else
    identifier = expr.type().get_string("tag");

  // Generate a tuple of the correct form for this type
  convert_identifier(identifier, expr.type(), bv);
  Z3_sort sort;
  create_type(expr.type(), sort);

  unsigned size = components.size();
  if (expr.id() == "union")
    size++;

  Z3_ast args[size];

  // Populate tuple with members of that struct/union
  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    convert_bv(expr.operands()[i], args[i]);
  }

  // Update unions "last-set" member to be the last field
  if (expr.id() == "union")
    args[size-1] = convert_number(i, config.ansi_c.int_width, false);

  // Create tuple itself and bind to sym name
  Z3_ast init_val = z3_api.mk_tuple(z3_ctx, sort, args, size);
  Z3_ast eq = Z3_mk_eq(z3_ctx, bv, init_val);
  // XXXjmorse - is this necessary?
  // We're generating a struct, not _actually_ binding it to a name.
  assert_formula(eq);

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_identifier_pointer

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_identifier_pointer(const exprt &expr, std::string symbol,
                                     Z3_ast &bv)
{
  DEBUGLOC;

  Z3_ast num;
  Z3_type_ast tuple_type;
  Z3_sort native_int_sort;
  std::string cte, identifier;
  unsigned int obj_num;

  if (int_encoding)
    native_int_sort = Z3_mk_int_sort(z3_ctx);
  else
    native_int_sort = Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  create_pointer_type(tuple_type);

  if (expr.get("value").compare("NULL") == 0 ||
      identifier == "0") {
    obj_num = pointer_logic.get_null_object();
  } else {
    // add object won't duplicate objs for identical exprs (it's a map)
    obj_num = pointer_logic.add_object(expr);
  }

  bv = z3_api.mk_var(z3_ctx, symbol.c_str(), tuple_type);

  // If this object hasn't yet been put in the address space record, we need to
  // assert that the symbol has the object ID we've allocated, and then fill out
  // the address space record.
  if (addr_space_data.find(obj_num) == addr_space_data.end()) {

    Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, tuple_type);

    Z3_ast args[2];
    args[0] = Z3_mk_int(z3_ctx, obj_num, native_int_sort);
    args[1] = Z3_mk_int(z3_ctx, 0, native_int_sort);

    Z3_ast ptr_val = Z3_mk_app(z3_ctx, decl, 2, args);
    Z3_ast constraint = Z3_mk_eq(z3_ctx, bv, ptr_val);
    assert_formula(constraint);

    typet ptr_loc_type = unsignedbv_typet(config.ansi_c.int_width);

    std::string start_name = "__ESBMC_ptr_obj_start_" + itos(obj_num);
    std::string end_name = "__ESBMC_ptr_obj_end_" + itos(obj_num);

    exprt start_sym = symbol_exprt(start_name, ptr_loc_type);
    exprt end_sym = symbol_exprt(end_name, ptr_loc_type);

    // Another thing to note is that the end var must be /the size of the obj/
    // from start. Express this in irep.
    std::string offs = integer2string(pointer_offset_size(expr.type()), 2);
    exprt const_offs_expr("constant", ptr_loc_type);
    const_offs_expr.set("value", offs);
    exprt start_plus_offs_expr("+", ptr_loc_type);
    start_plus_offs_expr.copy_to_operands(start_sym, const_offs_expr);
    exprt equality_expr("=", ptr_loc_type);
    equality_expr.copy_to_operands(end_sym, start_plus_offs_expr);

    // Also record the amount of memory space we're working with for later usage
    total_mem_space += pointer_offset_size(expr.type()).to_long() + 1;

    // Assert that start + offs == end
    Z3_ast offs_eq;
    convert_bv(equality_expr, offs_eq);
    assert_formula(offs_eq);

    // Even better, if we're operating in bitvector mode, it's possible that
    // Z3 will try to be clever and arrange the pointer range to cross the end
    // of the address space (ie, wrap around). So, also assert that end > start
    exprt wraparound_expr(">", ptr_loc_type);
    wraparound_expr.copy_to_operands(end_sym, start_sym);
    Z3_ast wraparound_eq;
    convert_bv(wraparound_expr, wraparound_eq);
    assert_formula(wraparound_eq);

    // We'll place constraints on those addresses later, in finalize_pointer_chain

    addr_space_data[obj_num] = pointer_offset_size(expr.type()).to_long() + 1;

    Z3_ast start_ast, end_ast;
    convert_bv(start_sym, start_ast);
    convert_bv(end_sym, end_ast);

    // Actually store into array
    Z3_ast range_tuple = z3_api.mk_var(z3_ctx,
                       ("__ESBMC_ptr_addr_range_" + itos(obj_num)).c_str(),
                       addr_space_tuple_sort);
    Z3_ast init_val = z3_api.mk_tuple(z3_ctx, addr_space_tuple_sort, start_ast,
                                      end_ast, NULL);
    Z3_ast eq = Z3_mk_eq(z3_ctx, range_tuple, init_val);
    assert_formula(eq);

    // Update array
    bump_addrspace_array(obj_num, range_tuple);
  }
}

/*******************************************************************
   Function: z3_convt::convert_zero_string

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_zero_string(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  // XXXjmorse - this method appears to just return a free variable. Surely
  // it should be selecting the zero_string field out of the referenced
  // string?
  Z3_type_ast array_type;

  create_array_type(expr.type(), array_type);

  bv = z3_api.mk_var(z3_ctx, "zero_string", array_type);
}

/*******************************************************************
   Function: z3_convt::convert_array

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_array(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  u_int i = 0;
  Z3_sort native_int_sort;
  Z3_type_ast array_type, elem_type;
  Z3_ast int_cte, val_cte;
  std::string value_cte;

  assert(expr.id() == "constant");
  native_int_sort = (int_encoding) ? Z3_mk_int_sort(z3_ctx)
                              : Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  create_type(expr.type().subtype(), elem_type);
  array_type = Z3_mk_array_type(z3_ctx, native_int_sort, elem_type);

 if (expr.type().subtype().id() == "struct")
    value_cte = "constant" + expr.op0().type().get_string("tag");
  else
    value_cte = expr.get_string("identifier") +
                expr.type().subtype().get("width").c_str();

  bv = z3_api.mk_var(z3_ctx, value_cte.c_str(), array_type);

  i = 0;
  forall_operands(it, expr)
  {
    int_cte = Z3_mk_int(z3_ctx, i, native_int_sort);

    convert_bv(*it, val_cte);

    bv = Z3_mk_store(z3_ctx, bv, int_cte, val_cte);
    ++i;
  }


  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_constant

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_constant(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  int64_t value;
  unsigned width;

  if (expr.type().id() == "c_enum") {
    // jmorse: value field of C enum type is in fact base 10, wheras everything
    // else is base 2.
    value = atol(expr.get_string("value").c_str());
  } else if (expr.type().id() == "bool")   {
    value = (expr.is_true()) ? 1 : 0;
  } else if (expr.type().id() == "pointer" && expr.get_string("value") ==
             "NULL")   {
    value = 0;
  } else if (is_signed(expr.type()))   {
    value = binary2integer(expr.get_string("value"), true).to_long();
  } else {
    value = binary2integer(expr.get_string("value"), false).to_long();
  }

  get_type_width(expr.type(), width);

  Z3_sort int_sort;
  if (int_encoding)
    int_sort = Z3_mk_int_sort(z3_ctx);
  else
    int_sort = Z3_mk_bv_type(z3_ctx, width);


  if (expr.type().id() == "unsignedbv") {
    bv = Z3_mk_unsigned_int64(z3_ctx, value, int_sort);
  } else if (expr.type().id() == "signedbv" || expr.type().id() == "c_enum") {
    bv = Z3_mk_int64(z3_ctx, value, int_sort);
  } else if (expr.type().id() == "fixedbv")    {

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
    convert_identifier_pointer(expr, itos(value), bv);
  } else if (expr.type().id() == "bool")     {
    if (expr.is_false())
      bv = Z3_mk_false(z3_ctx);
    else if (expr.is_true())
      bv = Z3_mk_true(z3_ctx);
  } else if (expr.type().id() == "array")     {
    convert_array(expr, bv);
  } else if (expr.type().id() == "struct" || expr.type().id() == "union") {
    convert_struct_union(expr, bv);
  }

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_bitwise

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_bitwise(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() >= 2);
  Z3_ast *args;
  unsigned i = 0, width;

  args = new Z3_ast[expr.operands().size() + 1];

  convert_bv(expr.op0(), args[0]);
  convert_bv(expr.op1(), args[1]);

  forall_operands(it, expr)
  {
    convert_bv(*it, args[i]);

    if (int_encoding) {
      get_type_width(expr.type(), width);
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
      throw new conv_error("bitwise ops not supported for type", expr);
  } else
    bv = args[i];

  delete[] args;
}

/*******************************************************************
   Function: z3_convt::convert_unary_minus

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_unary_minus(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast args[2];

  convert_bv(expr.op0(), args[0]);

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
}

/*******************************************************************
   Function: z3_convt::convert_if

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_if(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 3);

  Z3_ast operand0, operand1, operand2;

  convert_bv(expr.op0(), operand0);
  convert_bv(expr.op1(), operand1);
  convert_bv(expr.op2(), operand2);

  bv = Z3_mk_ite(z3_ctx, operand0, operand1, operand2);

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_logical_ops

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
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
    convert_bv(expr.op0(), bv);
  } else   {
    forall_operands(it, expr)
    {
      convert_bv(*it, args[i]);

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
}

/*******************************************************************
   Function: z3_convt::convert_logical_not

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_logical_not(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand0;

  convert_bv(expr.op0(), operand0);

  bv = Z3_mk_not(z3_ctx, operand0);

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_equality

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_equality(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  assert(expr.op0().type() == expr.op1().type());

  Z3_ast args[2];

  convert_bv(expr.op0(), args[0]);
  convert_bv(expr.op1(), args[1]);

  if (expr.id() == "=")
    bv = Z3_mk_eq(z3_ctx, args[0], args[1]);
  else
    bv = Z3_mk_distinct(z3_ctx, 2, args);

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_add_sub

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_pointer_arith(const exprt &expr, Z3_ast &bv)
{

  if (expr.operands().size() != 2)
    throw new conv_error("Pointer arithmatic takes two operands", expr);

  // So eight cases; one for each combination of two operands and the return
  // type, being pointer or nonpointer. So with P=pointer, N= notpointer,
  //    return    op1        op2        action
  //      N        N          N         Will never be fed here
  //      N        P          N         Expected arith option, then cast to int
  //      N        N          P            "
  //      N        P          P         Not permitted by C spec
  //      P        N          N         Return arith action with cast to pointer
  //      P        P          N         Calculate expected ptr arith operation
  //      P        N          P            "
  //      P        P          P         Not permitted by C spec
  //      NPP is the most dangerous - there's the possibility that an integer
  //      arithmatic is going to lead to an invalid pointer, that falls out of
  //      all dereference switch cases. So, we need to verify that all derefs
  //      have a finally case that asserts the val was a valid ptr XXXjmorse.
  int ret_is_ptr, op1_is_ptr, op2_is_ptr;
  ret_is_ptr = (expr.type().id() == "pointer") ? 4 : 0;
  op1_is_ptr = (expr.op0().type().id() == "pointer") ? 2 : 0;
  op2_is_ptr = (expr.op1().type().id() == "pointer") ? 1 : 0;

  const exprt *ptr_op, *non_ptr_op;
  switch (ret_is_ptr | op1_is_ptr | op2_is_ptr) {
    case 0:
      assert(false);
      break;
    case 3:
    case 7:
      throw new conv_error("Pointer arithmatic with two pointer operands",expr);
      break;
    case 4:
      // Artithmatic operation that has the result type of ptr.
      // Should have been handled at a higher level
      throw new conv_error("Non-pointer op being interpreted as pointer without"
                           " typecast", expr);
      break;
    case 1:
    case 2:
      { // Required to give a variable lifetime to the cast/add variables
      ptr_op = (op1_is_ptr) ? &expr.op0() : &expr.op1();
      non_ptr_op = (op1_is_ptr) ? &expr.op1() : &expr.op0();

      exprt add("+", ptr_op->type());
      add.copy_to_operands(*ptr_op, *non_ptr_op);
      // That'll generate the correct pointer arithmatic; now typecast
      exprt cast("typecast", expr.type());
      cast.copy_to_operands(add);
      convert_bv(cast, bv);
      break;
      }
    case 5:
    case 6:
      {
      ptr_op = (op1_is_ptr) ? &expr.op0() : &expr.op1();
      non_ptr_op = (op1_is_ptr) ? &expr.op1() : &expr.op0();

      // Actually perform some pointer arith
      std::string type_size =
        integer2string(pointer_offset_size(ptr_op->type().subtype()), 2);

      exprt mul("*", signedbv_typet(config.ansi_c.int_width));
      exprt constant("constant", unsignedbv_typet(config.ansi_c.int_width));
      constant.set("value", type_size);
      mul.copy_to_operands(*non_ptr_op, constant);

      // Add or sub that value
      exprt ptr_offset("pointer_offset", unsignedbv_typet(config.ansi_c.int_width));
      ptr_offset.copy_to_operands(*ptr_op);
      exprt arith_op(expr.id().as_string(), unsignedbv_typet(config.ansi_c.int_width));
      arith_op.copy_to_operands(ptr_offset, mul);

      // Voila, we have our pointer arithmatic
      convert_bv(arith_op, bv);

      // That calculated the offset; update field in pointer.
      Z3_ast the_ptr;
      convert_bv(*ptr_op, the_ptr);
      bv = z3_api.mk_tuple_update(z3_ctx, the_ptr, 1, bv);

      break;
      }
  }
}

void
z3_convt::convert_add_sub(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  typedef Z3_ast (*call1)(Z3_context, Z3_ast, Z3_ast);
  typedef Z3_ast (*call2)(Z3_context, unsigned int, const Z3_ast *);

  call1 bvcall;
  call2 intcall;

  if (expr.type().id() == "pointer" || expr.op0().type().id() == "pointer" ||
      expr.op1().type().id() == "pointer") {
    convert_pointer_arith(expr, bv);
    return;
  }

  if (expr.id() == "+") {
    bvcall = Z3_mk_bvadd;
    intcall = Z3_mk_add;
  } else if (expr.id() == "-") {
    bvcall = Z3_mk_bvsub;
    intcall = Z3_mk_sub;
  } else {
    throw new conv_error("Non add/sub expr in convert_add_sub", expr);
  }

  assert(expr.operands().size() >= 2);
  Z3_ast *args, accuml;
  u_int i = 0, size;

  size = expr.operands().size();
  args = new Z3_ast[size];

  forall_operands(it, expr)
  {
    convert_bv(*it, args[i]);
    ++i;
  }

  if (!int_encoding) {
    accuml = args[0];
    for (i = 1; i < size; i++) {
      accuml = bvcall(z3_ctx, accuml, args[i]);
    }
  } else {
    accuml = intcall(z3_ctx, i, args);
  }

  bv = accuml;

  delete[] args;

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_div

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_div(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  if (expr.type().id() == "pointer" || expr.op0().type().id() == "pointer" ||
      expr.op1().type().id() == "pointer") {
    throw new conv_error("Pointer operands to mod are not permitted", expr);
  }

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;

  convert_bv(expr.op0(), operand0);
  convert_bv(expr.op1(), operand1);

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
    } else {
      throw new conv_error("unsupported type for /: ", expr);
    }
  }
}

/*******************************************************************
   Function: z3_convt::convert_mod

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_mod(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  if (expr.type().id() == "pointer" || expr.op0().type().id() == "pointer" ||
      expr.op1().type().id() == "pointer") {
    throw new conv_error("Pointer operands to divide are not permitted", expr);
  }

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;

  convert_bv(expr.op0(), operand0);
  convert_bv(expr.op1(), operand1);

  if (expr.type().id() != "signedbv" && expr.type().id() != "unsignedbv")
    throw new conv_error("unsupported type for mod", expr);

  if (int_encoding) {
    bv = Z3_mk_mod(z3_ctx, operand0, operand1);
  } else   {
    if (expr.type().id() == "signedbv") {
      bv = Z3_mk_bvsrem(z3_ctx, operand0, operand1);
    }   else if (expr.type().id() == "unsignedbv") {
      bv = Z3_mk_bvurem(z3_ctx, operand0, operand1);
    }
  }
}

/*******************************************************************
   Function: z3_convt::convert_mul

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_mul(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  if (expr.type().id() == "pointer" || expr.op0().type().id() == "pointer" ||
      expr.op1().type().id() == "pointer") {
    throw new conv_error("Pointer operands to mod are not permitted", expr);
  }

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

  forall_operands(it, expr)
  {
    convert_bv(*it, args[i]);

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

  if (expr.type().id() == "fixedbv" && !int_encoding) {
    fixedbvt fbt(expr);
    bv =
      Z3_mk_extract(z3_ctx, fbt.spec.width + fraction_bits - 1, fraction_bits,
                    bv);
  }

  delete[] args;

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_pointer

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_address_of(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  assert(expr.type().id() == "pointer");

  Z3_type_ast pointer_type;
  Z3_ast offset;
  std::string symbol_name, out;

  create_pointer_type(pointer_type);
  offset = convert_number(0, config.ansi_c.int_width, true);

  if (expr.op0().id() == "index") {
    // Borrow an idea from CBMC; instead of munging all the pointer arith
    // here, we instead take the address of the base object and add to it
    // just by generating th erelevant ireps.

    exprt obj = expr.op0().op0();
    exprt idx = expr.op0().op1();

    // Pick pointer-to array subtype; need to make pointer arith work.
    exprt addrof("address_of", pointer_typet(obj.type().subtype()));
    addrof.copy_to_operands(obj);
    exprt plus("+", addrof.type());
    plus.copy_to_operands(addrof, idx);
    convert_bv(plus, bv);
  } else if (expr.op0().id() == "member") {
    const member_exprt &member_expr = to_member_expr(expr.op0());

    if (member_expr.op0().type().id() == "struct" ||
        member_expr.op0().type().id() == "union") {
      const struct_typet &struct_type =to_struct_type(member_expr.op0().type());
      const irep_idt component_name = member_expr.get_component_name();

      int64_t offs = member_offset(struct_type, component_name).to_long();

      exprt addrof("address_of", pointer_typet(member_expr.op0().type()));
      addrof.copy_to_operands(member_expr.op0());
      convert_bv(addrof, bv);

      // Update pointer offset to offset to that field.
      Z3_ast num = convert_number(offs, config.ansi_c.int_width, true);
      bv = z3_api.mk_tuple_update(z3_ctx, bv, 1, num);
    } else {
      throw new conv_error("Non struct/union operand to member", expr);
    }
  } else if (expr.op0().id() == "symbol" || expr.op0().id() == "code") {
    convert_identifier_pointer(expr.op0(), expr.op0().get_string("identifier"),
                               bv);
  } else if (expr.op0().id() == "string-constant") {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    std::string identifier = "address_of_str_const(" +
                             expr.op0().get_string("value") + ")";
    convert_identifier_pointer(expr.op0(), identifier, bv);
  } else {
    throw new conv_error("Unrecognized address_of operand", expr);
  }

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_array_of

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_array_of(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.type().id() == "array");
  assert(expr.operands().size() == 1);

  Z3_ast value, index;
  Z3_type_ast array_type = 0;
  const array_typet &array_type_size = to_array_type(expr.type());
  std::string tmp, out, identifier;
  int64_t size;
  u_int j;
  static u_int inc = 0; // valid-ish as static, should be member.
  unsigned width;

  size = binary2integer(array_type_size.size().get_string("value"), false)
                        .to_long();

  size = (size == 0) ? 100 : size; //fill in at least one position

  convert_bv(expr.op0(), value);

  create_array_type(expr.type(), array_type);
  get_type_width(expr.op0().type(), width);

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
}

/*******************************************************************
   Function: z3_convt::convert_index

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_index(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);

  Z3_ast args[2];
  std::string identifier;

  convert_bv(expr.op0(), args[0]);
  convert_bv(expr.op1(), args[1]);

  // XXXjmorse - consider situation where a pointer is indexed. Should it
  // give the address of ptroffset + (typesize * index)?
  bv = Z3_mk_select(z3_ctx, args[0], args[1]);

  DEBUGLOC;
}

/*******************************************************************
   Function: z3_convt::convert_shift

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_shift(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 2);
  Z3_ast operand0, operand1;
  unsigned width_expr, width_op0, width_op1;

  convert_bv(expr.op0(), operand0);
  convert_bv(expr.op1(), operand1);

  get_type_width(expr.type(), width_expr);
  get_type_width(expr.op0().type(), width_op0);
  get_type_width(expr.op1().type(), width_op1);

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
    throw new conv_error("Non-shift operation in convert_shift", expr);

  if (int_encoding) {
    if (expr.type().id() == "signedbv")
      bv = Z3_mk_bv2int(z3_ctx, bv, true);
    else if (expr.type().id() == "unsignedbv")
      bv = Z3_mk_bv2int(z3_ctx, bv, false);
    else
      throw new conv_error("No bitshift ops for expr type", expr);
  }
}

/*******************************************************************
   Function: z3_convt::convert_abs

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_abs(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  unsigned width;
  //std::string out;

  get_type_width(expr.type(), width);

  const exprt::operandst &operands = expr.operands();

  if (operands.size() != 1)
    throw new conv_error("abs takes one operand", expr);

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
  } else {
    throw new conv_error("Unexpected type in convert_abs", expr);
  }

  convert_bv(op0, operand[0]);

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
}

/*******************************************************************
   Function: z3_convt::convert_with

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_with(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 3);
  Z3_ast array_var, array_val, operand0, operand1, operand2;

  if (expr.type().id() == "struct" || expr.type().id() == "union") {
    unsigned int idx;

    convert_bv(expr.op0(), array_var);
    convert_bv(expr.op2(), array_val);

    idx = convert_member_name(expr.op0(), expr.op1());
    bv = z3_api.mk_tuple_update(z3_ctx, array_var, idx, array_val);

    // Update last-updated-field field if it's a union
    if (expr.type().id() == "union") {
       unsigned int components_size =
                  to_struct_union_type( expr.op0().type()).components().size();
       bv = z3_api.mk_tuple_update(z3_ctx, bv, components_size,
                              convert_number(idx, config.ansi_c.int_width, 0));
    }
  } else if (expr.type().id() == "array") {

    convert_bv(expr.op0(), operand0);
    convert_bv(expr.op1(), operand1);
    convert_bv(expr.op2(), operand2);

    bv = Z3_mk_store(z3_ctx, operand0, operand1, operand2);
  } else {
    throw new conv_error("with applied to non-struct/union/array obj", expr);
  }
}

/*******************************************************************
   Function: z3_convt::convert_bitnot

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_bitnot(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand0;

  convert_bv(expr.op0(), operand0);

  if (int_encoding)
    bv = Z3_mk_not(z3_ctx, operand0);
  else
    bv = Z3_mk_bvnot(z3_ctx, operand0);

  DEBUGLOC;
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
  u_int i = 0;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (it->get("name").compare(rhs.get_string("component_name")) == 0)
      return i;
  }

  throw new conv_error("component name not found in struct", lhs);
}

/*******************************************************************
   Function: z3_convt::select_pointer_offset

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::select_pointer_offset(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast pointer;

  convert_bv(expr.op0(), pointer);

  bv = z3_api.mk_tuple_select(z3_ctx, pointer, 1); //select pointer offset
}

/*******************************************************************
   Function: z3_convt::convert_member

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_member(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  u_int j = 0;
  Z3_ast struct_var;

  convert_bv(expr.op0(), struct_var);

  j = convert_member_name(expr.op0(), expr);

  if (expr.op0().type().id() == "union") {
    union_varst::const_iterator cache_result = union_vars.find(
      expr.op0().get_string("identifier").c_str());
    if (cache_result != union_vars.end()) {
      bv = z3_api.mk_tuple_select(z3_ctx, struct_var, cache_result->second);
      return;
    }
  }

  bv = z3_api.mk_tuple_select(z3_ctx, struct_var, j);

  DEBUGLOC;
}

/*******************************************************************
   Function: convert_pointer_object

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_pointer_object(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1 && is_ptr(expr.op0().type()));

  convert_bv(expr.op0(), bv);
  bv = z3_api.mk_tuple_select(z3_ctx, bv, 0);
}

/*******************************************************************
   Function: z3_convt::convert_zero_string_length

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_zero_string_length(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);
  Z3_ast operand;

  convert_bv(expr.op0(), operand);

  bv = z3_api.mk_tuple_select(z3_ctx, operand, 0);
}

/*******************************************************************
   Function: z3_convt::convert_byte_update

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_byte_update(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(!int_encoding && "byte operation in integer encoding mode is invalid");

  assert(expr.operands().size() == 3);
  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  mp_integer i;
  if (to_integer(expr.op1(), i)) {
    convert_bv(expr.op0(), bv);
    return;
  }

  const typet *conv_type;
  Z3_ast orig_val, update_value;

  // Generate bitvector of the datastructure we'll be dealing with. This needs to
  // be extracted from the fact that the points-to analysis gives us an expr for
  // the first element the pointer we were dealing with points at.
  if (expr.op0().id() == "member" || expr.op0().id() == "index") {
    // extract the actual object or struct from this expression.
    const exprt &theobj = fetch_base_object(expr.op0());
    convert_bv(theobj, orig_val);
    conv_type = &theobj.type();
  } else {
    // Or, it's just some entirely usable object.
    convert_bv(expr.op0(), orig_val);
    conv_type = &expr.op0().type();
  }
  orig_val = to_bv(*conv_type, orig_val);

  convert_bv(expr.op2(), update_value);

  uint width_op2;
  get_type_width(expr.op2().type(), width_op2);

  // Irritatingly, there's no way of performing a bit update, so instead extract
  // the bits either side of the portion we want, and concatonate it all.
  uint64_t upper, lower;
  unsigned int width;

  width = Z3_get_bv_sort_size(z3_ctx, Z3_get_sort(z3_ctx, orig_val));
  lower = i.to_long() * 8;
  upper = lower + width_op2 - 1;

  Z3_ast lowerbv = NULL, upperbv = NULL, updatedval;
  if (lower != 0)
    lowerbv = Z3_mk_extract(z3_ctx, lower-1, 0, orig_val);
  if (upper != 0)
    upperbv = Z3_mk_extract(z3_ctx, width-1, upper+1, orig_val);

  updatedval = update_value;
  if (lowerbv != NULL)
    updatedval = Z3_mk_concat(z3_ctx, lowerbv, update_value);
  if (upperbv != NULL)
    updatedval = Z3_mk_concat(z3_ctx, updatedval, upperbv);

  bv = from_bv(expr.type(), updatedval, NULL);
}

/*******************************************************************
   Function: z3_convt::convert_byte_extract

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_byte_extract(const exprt &expr, Z3_ast &bv)
{
  Z3_ast op0;
  const typet *conv_type;
  unsigned width;

  get_type_width(expr.type(), width);
  if (expr.op0().id() == "member" || expr.op0().id() == "index") {
    // the point-to analysis hands us the first /thing/ in an object, so that
    // the address of it gives the address of the object?
    // Thus, extract the actual object or struct from that expression.
    const exprt &theobj = fetch_base_object(expr.op0());
    convert_bv(theobj, op0);
    conv_type = &theobj.type();
  } else {
    // Or, it's just some entirely usable object.
    convert_bv(expr.op0(), op0);
    conv_type = &expr.op0().type();
  }
  op0 = to_bv(*conv_type, op0);

  assert(!int_encoding && "byte operation in integer encoding mode is invalid");
  assert(expr.operands().size() == 2);
  // op0 is object to extract from
  // op1 is byte field to extract from.

  mp_integer i;
  if (to_integer(expr.op1(), i)) {
    // Non-constant byte extract. Shift symbolic distance and extract the lower
    // portion of the bit vector.
    Z3_sort offs_sort;
    Z3_ast byteoffset, bitoffset, full_bitoffset, eight, shifted;
    convert_bv(expr.op1(), byteoffset);

    // Multiply by 8, then coerce shift value to be same width as op0, for shrs
    // purpose. We don't have to worry about overflow here: if it's big enough
    // to overflow in the minimum size (8 bits), or indeed any other size, then
    // it should violate the pointer bound checks introduced by dereference.
    offs_sort = Z3_get_sort(z3_ctx, byteoffset);
    eight = Z3_mk_unsigned_int(z3_ctx, 8, offs_sort);
    bitoffset = Z3_mk_bvmul(z3_ctx, byteoffset, eight);
    // Now for the extension,
    unsigned int width = Z3_get_bv_sort_size(z3_ctx, Z3_get_sort(z3_ctx, op0));
    width -= Z3_get_bv_sort_size(z3_ctx, Z3_get_sort(z3_ctx, bitoffset));
    full_bitoffset = Z3_mk_zero_ext(z3_ctx, width, bitoffset);
    shifted = Z3_mk_bvlshr(z3_ctx, op0, full_bitoffset);

    // And after all that, pick out the part of the data that we want.
    bv = Z3_mk_extract(z3_ctx, width-1, 0, shifted);
  } else {
    // Constant byte extract. Pick the byte range and extract it.
    uint64_t upper, lower;

    lower = i.to_long() * 8;
    upper = lower + width - 1;

    bv = Z3_mk_extract(z3_ctx, upper, lower, op0);
  }
}

Z3_ast z3_convt::to_bv(const typet &type, Z3_ast src)
{

  // XXXjmorse - endianness concerns?
  if (type.id() == "struct")
    return struct_to_bv(type, src);
  else if (type.id() == "union")
    return union_to_bv(type, src);
  else if (type.id() == "array")
    return array_to_bv(type, 0, 0, src);
  else if (type.id() == "signedbv" || type.id() == "unsignedbv" ||
           type.id() == "fixedbv")
    return src; // Should already be in a bitvector format.
  else if (type.id() == "bool") {
    // XXXjmorse - here we're defining a boolean to be a single bit in an 8 bit
    // integer. This is something that should be configurable from the top level.
    Z3_sort sort = Z3_mk_bv_sort(z3_ctx, 8);
    Z3_ast true_bit = Z3_mk_unsigned_int(z3_ctx, 1, sort);
    Z3_ast false_bit = Z3_mk_unsigned_int(z3_ctx, 0, sort);
    Z3_ast result = Z3_mk_ite(z3_ctx, src, true_bit, false_bit);
    return result;
  } else
    throw new conv_error("unsupported to-bitvector conversion type", type);
}

Z3_ast z3_convt::struct_to_bv(const typet &type, Z3_ast src)
{
  // Convert input struct to a bit vector, using src, which should be a tuple.

  Z3_ast chain = NULL;
  unsigned int i;
  const struct_typet &struct_type = to_struct_type(type);
  const struct_typet::componentst &components = struct_type.components();

  for (i = 0; i < components.size(); i++)
  {
    Z3_ast tmp;

    // Extract each member from the tuple; concatenate.
    tmp = z3_api.mk_tuple_select(z3_ctx, src, i);

    if (chain == NULL)
      chain = tmp;
    else
      chain = Z3_mk_concat(z3_ctx, chain, tmp);
  }

  return chain;
}

Z3_ast z3_convt::union_to_bv(const typet &type, Z3_ast src)
{

  std::cerr << "Union <=> bitvector byte operations currently unsupported";
  std::cerr << std::endl;
  abort();
}

Z3_ast z3_convt::array_to_bv(const typet &type, unsigned int startidx,
                             unsigned int endidx, Z3_ast src)
{

  if (startidx == 0 && endidx == 0) {
    // Determine size - I'm assuming it's in elements.
    exprt &size = (exprt&)type.find("size");
    assert(size != get_nil_irep());
    simplify(size); // Inefficient; should be simplified earlier.
    assert(size.id() == "constant");
    unsigned int tmp = binary2integer(size.get("value").as_string(), false).to_long();
    endidx = tmp - 1;
  }

  // Here startidx and endidx indicate how much of the array we wish to extract.
  unsigned int i;
  Z3_ast chain = NULL;
  for (i = startidx; i <= endidx; i++) {
    Z3_ast tmp;
    Z3_ast idx = convert_number(i, config.ansi_c.int_width, true);
    tmp = Z3_mk_select(z3_ctx, src, idx);

    if (chain == NULL)
      chain = tmp;
    else
      chain = Z3_mk_concat(z3_ctx, chain, tmp);
  }

  return chain;
}

Z3_ast z3_convt::from_bv(const typet &type, Z3_ast src, Z3_ast orig)
{

  // XXXjmorse - endianness concerns.
  if (type.id() == "struct")
    return struct_from_bv(type, src);
  else if (type.id() == "union")
    return union_from_bv(type, src);
  else if (type.id() == "array")
    return array_from_bv(type, 0, 0, src, orig);
  else if (type.id() == "signedbv" || type.id() == "unsignedbv" ||
           type.id() == "fixedbv")
    return src; // Should already be in a bitvector format.
  else if (type.id() == "bool") {
    // XXXjmorse - here we're defining a boolean to be a single bit in an 8 bit
    // integer. This is something that should be configurable from the top level.
    Z3_sort sort = Z3_mk_bv_sort(z3_ctx, 8);
    Z3_ast false_bit = Z3_mk_unsigned_int(z3_ctx, 0, sort);
    Z3_ast eq = Z3_mk_eq(z3_ctx, false_bit, src);
    Z3_ast result = Z3_mk_ite(z3_ctx, eq, Z3_mk_true(z3_ctx),Z3_mk_false(z3_ctx));
    return result;
  } else
    throw new conv_error("unsupported from-bitvector conversion type", type);
}

Z3_ast z3_convt::struct_from_bv(const typet &type, Z3_ast src)
{
  const struct_typet &struct_type = to_struct_type(type);
  const struct_typet::componentst &components = struct_type.components();

  Z3_sort our_struct_type;
  create_struct_union_type(type, false, our_struct_type);

  unsigned int size = components.size(), offset = 0, idx = 0;
  Z3_ast args[size];
  for (struct_typet::componentst::const_iterator it = components.begin();
       it != components.end(); it++)
  {
    unsigned width;
    const typet &item_type = struct_type.component_type(it->get_name());
    get_type_width(item_type, width);

    Z3_ast bv = Z3_mk_extract(z3_ctx, offset, offset + width - 1, src);
    offset += width;
    Z3_ast item = from_bv(item_type, bv, NULL);
    args[idx++] = item;
  }

  return z3_api.mk_tuple(z3_ctx, our_struct_type, args, size);
}

Z3_ast z3_convt::union_from_bv(const typet &type, Z3_ast src)
{

  std::cerr << "Converting to union from bitvector currently unsupported";
  std::cerr << std::endl;
  abort();
}

Z3_ast z3_convt::array_from_bv(const typet &type, unsigned int startidx,
                               unsigned int endidx, Z3_ast src, Z3_ast orig)
{

  // Again, if no size given, assume whole array.
  if (startidx == 0 && endidx == 0) {
    exprt &size = (exprt&)type.find("size");
    assert(size != get_nil_irep());
    simplify(size); // Inefficient; should be simplified earlier.
    assert(size.id() == "constant");
    unsigned int tmp = binary2integer(size.get("value").as_string(), false).to_long();
    endidx = tmp - 1;

    // Also, create a unique new array to update all the elements into.
    Z3_sort array_sort;
    create_type(type, array_sort);
    orig = Z3_mk_fresh_const(z3_ctx, NULL, array_sort);
  }

  unsigned int width;
  get_type_width(type.subtype(), width);

  // Loop through bit vector, extracting elements and updating.
  Z3_ast chain = NULL;
  unsigned int offset = 0, i = 0;
  for (i = startidx; i <= endidx; i++) {
    Z3_ast tmp;
    Z3_ast val = Z3_mk_extract(z3_ctx, offset + width - 1, offset, src);
    offset += width;

    orig = Z3_mk_store(z3_ctx, orig,
                       convert_number(i, config.ansi_c.int_width, true),
                       val);
  }

  return orig;
}
/*******************************************************************
   Function: z3_convt::convert_isnan

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_isnan(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  assert(expr.operands().size() == 1);

  const typet &op_type = expr.op0().type();

  if (op_type.id() == "fixedbv") {
    Z3_ast op0;
    unsigned width;

    get_type_width(op_type, width);

    convert_bv(expr.op0(), op0);

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
  } else {
    throw new conv_error("isnan with unsupported operand type", expr);
  }
}

/*******************************************************************
   Function: z3_convt::convert_z3_expr

   Inputs:

   Outputs:

   Purpose:

 \*******************************************************************/

void
z3_convt::convert_z3_expr(const exprt &expr, Z3_ast &bv)
{
  DEBUGLOC;

  irep_idt exprid = expr.id();

  if (exprid == "symbol")
    convert_identifier(expr.get_string("identifier"), expr.type(), bv);
  else if (exprid == "nondet_symbol")
    convert_identifier("nondet$" + expr.get_string("identifier"),
                       expr.type(), bv);
  else if (exprid == "typecast")
    convert_typecast(expr, bv);
  else if (exprid == "struct" || exprid == "union")
    convert_struct_union(expr, bv);
  else if (exprid == "constant")
    convert_constant(expr, bv);
  else if (exprid == "bitand" || exprid == "bitor" || exprid ==
           "bitxor"
           || exprid == "bitnand" || exprid == "bitnor" || exprid ==
           "bitnxor")
    convert_bitwise(expr, bv);
  else if (exprid == "bitnot")
    convert_bitnot(expr, bv);
  else if (exprid == "unary-")
    convert_unary_minus(expr, bv);
  else if (exprid == "if")
    convert_if(expr, bv);
  else if (exprid == "and" || exprid == "or" || exprid == "xor")
    convert_logical_ops(expr, bv);
  else if (exprid == "not")
    convert_logical_not(expr, bv);
  else if (exprid == "=" || exprid == "notequal")
    convert_equality(expr, bv);
  else if (exprid == "<=" || exprid == "<" || exprid == ">="
           || exprid == ">")
    bv = convert_cmp(expr);
  else if (exprid == "+" || exprid == "-")
    convert_add_sub(expr, bv);
  else if (exprid == "/")
    convert_div(expr, bv);
  else if (exprid == "mod")
    convert_mod(expr, bv);
  else if (exprid == "*")
    convert_mul(expr, bv);
  else if (exprid == "address_of" || exprid == "implicit_address_of"
           || exprid == "reference_to")
    return convert_address_of(expr, bv);
  else if (exprid == "array_of")
    convert_array_of(expr, bv);
  else if (exprid == "index")
    convert_index(expr, bv);
  else if (exprid == "ashr" || exprid == "lshr" || exprid == "shl")
    convert_shift(expr, bv);
  else if (exprid == "abs")
    convert_abs(expr, bv);
  else if (exprid == "with")
    convert_with(expr, bv);
  else if (exprid == "member")
    convert_member(expr, bv);
  else if (exprid == "zero_string")
    convert_zero_string(expr, bv);
  else if (exprid == "pointer_offset")
    select_pointer_offset(expr, bv);
  else if (exprid == "pointer_object")
    convert_pointer_object(expr, bv);
  else if (exprid == "same-object")
    bv = convert_same_object(expr);
  else if (exprid == "string-constant") {
    exprt tmp;
    string2array(expr, tmp);
    convert_bv(tmp, bv);
  } else if (exprid == "zero_string_length")
    convert_zero_string_length(expr.op0(), bv);
  else if (exprid == "replication")
    assert(expr.operands().size() == 2);
  else if (exprid == "is_dynamic_object")
    bv = convert_is_dynamic_object(expr);
  else if (exprid == "byte_update_little_endian" ||
           exprid == "byte_update_big_endian")
    convert_byte_update(expr, bv);
  else if (exprid == "byte_extract_little_endian" ||
           exprid == "byte_extract_big_endian")
    convert_byte_extract(expr, bv);
#if 1
  else if (exprid == "isnan")
    convert_isnan(expr, bv);
#endif
  else if (exprid == "width")
    bv = convert_width(expr);
  else if (exprid == "overflow-+" || exprid == "overflow--" || exprid == "overflow-*")
    bv = convert_overflow_sum_sub_mul(expr);
  else if (has_prefix(expr.id_string(), "overflow-typecast-"))
    bv = convert_overflow_typecast(expr);
  else if (exprid == "overflow-unary-")
    bv = convert_overflow_unary(expr);
  else if (exprid == "memory-leak")
    bv = convert_memory_leak(expr);
  else 
    throw new conv_error("Unrecognized expression type", expr);
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



  try {
    if (expr.id() == "=" && value) {
      assert(expr.operands().size() == 2);

      Z3_ast result, operand[2];
      const exprt &op0 = expr.op0();
      const exprt &op1 = expr.op1();

      if (assign_z3_expr(expr) && ignoring_expr) {
        convert_bv(op0, operand[0]);
        convert_bv(op1, operand[1]);

        if (op0.type().id() == "pointer" && op1.type().id() == "pointer") {
          Z3_ast pointer[2], formula[2];

          pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 0);
          pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 0);

          formula[0] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);
          pointer[0] = z3_api.mk_tuple_select(z3_ctx, operand[0], 1);

          pointer[1] = z3_api.mk_tuple_select(z3_ctx, operand[1], 1);
          formula[1] = Z3_mk_eq(z3_ctx, pointer[0], pointer[1]);

          if (expr.op0().type().id() == "bool")
            result = Z3_mk_iff(z3_ctx, formula[0], formula[1]);
          else
            result = Z3_mk_and(z3_ctx, 2, formula);

          assert_formula(result);
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

          assert_formula(result);

          if (z3_prop.uw && expr.op0().get_string("identifier").find("guard_exec") !=
              std::string::npos
              && z3_prop.assumpt.size() < max_core_size) {
            if (!op1.is_true())
              generate_assumptions(expr, operand[0]);
          }
        }
      }
    }
    ignoring_expr = true;
  } catch (conv_error *e) {
    std::cerr << e->to_string() << std::endl;
    ignoring(expr);
  }

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

void
z3_convt::get_type_width(const typet &t, unsigned &width)
{

  if (boolbv_get_width(t, width))
    throw new z3_convt::conv_error("Failed to determine type bitwidth", t);

  return;
}

Z3_context z3_convt::z3_ctx = NULL;

unsigned int z3_convt::num_ctx_ileaves = 0;
bool z3_convt::s_is_uw = false;
