/*******************************************************************
   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <assert.h>
#include <ctype.h>
#include <malloc.h>
#include <fstream>
#include <sstream>
#include <std_expr.h>
#include <irep2.h>
#include <migrate.h>
#include <arith_tools.h>
#include <std_types.h>
#include <config.h>
#include <i2string.h>
#include <expr_util.h>
#include <string2array.h>
#include <pointer_offset_size.h>
#include <find_symbols.h>
#include <prefix.h>
#include <fixedbv.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

#define cast_to_z3(arg) reinterpret_cast<Z3_ast &>((arg))

static std::vector<Z3_ast> core_vector;
static u_int unsat_core_size = 0;
static u_int assumptions_status = 0;

extern void finalize_symbols(void);

z3_convt::~z3_convt()
{

  if (model != NULL)
    Z3_del_model(z3_ctx, model);

  if (smtlib) {
    std::ofstream temp_out;
    Z3_string smt_lib_str, logic;
    Z3_ast *assumpt_array =
	          (Z3_ast *)alloca((assumpt.size() + 1) * sizeof(Z3_ast));
    Z3_ast formula;
    formula = Z3_mk_true(z3_ctx);

    std::list<Z3_ast>::const_iterator it;
    unsigned int i;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumpt_array[i] = *it;
    }

    if (int_encoding)
      logic = "QF_AUFLIRA";
    else
      logic = "QF_AUFBV";

    smt_lib_str = Z3_benchmark_to_smtlib_string(z3_ctx, "ESBMC", logic,
                                    "unknown", "", assumpt.size(),
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
#ifndef _WIN32
    // This call is an undocumented internal api of Z3's: it causes Z3 to free its
    // internal symbol table, which it otherwise doesn't, leading to vast
    // quantities of leaked memory. This will stop work/linking when Microsoft
    // eventually work out they should be stripping the linux binaries they
    // release.
    // Unfortnately it doesn't work like that on Windows, so only try this if
    // we're on another platform. And hope that perhaps Windows' Z3_reset_memory
    // works as advertised.
    finalize_symbols();
#endif
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

  // Place locations of numerical addresses for null and invalid_obj.

  Z3_ast tmp = z3_api.mk_var("__ESBMC_ptr_obj_start_0", native_int_sort);
  Z3_ast num = convert_number(0, config.ansi_c.int_width, true);
  Z3_ast eq = Z3_mk_eq(z3_ctx, tmp, num);
  assert_formula(eq);

  tmp = z3_api.mk_var("__ESBMC_ptr_obj_end_0", native_int_sort);
  num = convert_number(0, config.ansi_c.int_width, true);
  eq = Z3_mk_eq(z3_ctx, tmp, num);
  assert_formula(eq);

  tmp = z3_api.mk_var("__ESBMC_ptr_obj_start_1", native_int_sort);
  num = convert_number(1, config.ansi_c.int_width, true);
  eq = Z3_mk_eq(z3_ctx, tmp, num);
  assert_formula(eq);

  tmp = z3_api.mk_var("__ESBMC_ptr_obj_end_1", native_int_sort);
  num = convert_number(0xFFFFFFFFFFFFFFFFULL,
                              config.ansi_c.int_width, true);
  eq = Z3_mk_eq(z3_ctx, tmp, num);
  assert_formula(eq);

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

  num = convert_number(0, config.ansi_c.int_width, true);
  Z3_ast initial_val = z3_api.mk_tuple(addr_space_tuple_sort, num, num, NULL);

  Z3_ast initial_const = Z3_mk_const_array(z3_ctx, native_int_sort, initial_val);
  Z3_ast first_name = z3_api.mk_var("__ESBMC_addrspace_arr_0",
                                    addr_space_arr_sort);
  eq = Z3_mk_eq(z3_ctx, first_name, initial_const);
  assert_formula(eq);

  Z3_ast range_tuple = z3_api.mk_var("__ESBMC_ptr_addr_range_0",
                                     addr_space_tuple_sort);
  initial_val = z3_api.mk_tuple(addr_space_tuple_sort, num, num, NULL);
  eq = Z3_mk_eq(z3_ctx, initial_val, range_tuple);
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.get_null_object(), range_tuple);

  // We also have to initialize the invalid object... however, I've no idea
  // what it /means/ yet, so go for some arbitary value.
  num = convert_number(1, config.ansi_c.int_width, true);
  range_tuple = z3_api.mk_var("__ESBMC_ptr_addr_range_1",
                              addr_space_tuple_sort);
  initial_val = z3_api.mk_tuple(addr_space_tuple_sort, num, num, NULL);
  eq = Z3_mk_eq(z3_ctx, initial_val, range_tuple);
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.get_invalid_object(), range_tuple);

  // Associate the symbol "0" with the null object; this is necessary because
  // of the situation where 0 is valid as a representation of null, but the
  // frontend (for whatever reasons) converts it to a symbol rather than the
  // way it handles NULL (constant with val "NULL")
  Z3_sort pointer_type;
  create_pointer_type(pointer_type);
  Z3_ast zero_sym = z3_api.mk_var("0", pointer_type);
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(z3_ctx, pointer_type);

  Z3_ast args[2];
  args[0] = Z3_mk_int(z3_ctx, 0, native_int_sort);
  args[1] = Z3_mk_int(z3_ctx, 0, native_int_sort);
  Z3_ast ptr_val = Z3_mk_app(z3_ctx, decl, 2, args);
  Z3_ast constraint = Z3_mk_eq(z3_ctx, zero_sym, ptr_val);
  assert_formula(constraint);

  // Do the same thing, for the name "NULL".
  Z3_ast null_sym = z3_api.mk_var("NULL", pointer_type);
  constraint = Z3_mk_eq(z3_ctx, null_sym, ptr_val);
  assert_formula(constraint);

  // And for the "INVALID" object (which we're issuing with a name now), have
  // a pointer object num of 1, and a free pointer offset. Anything of worth
  // using this should extract only the object number.

  args[0] = Z3_mk_int(z3_ctx, 1, native_int_sort);
  args[1] = Z3_mk_fresh_const(z3_ctx, NULL, pointer_type);
  Z3_ast invalid = z3_api.mk_tuple_update(args[1], 0, args[0]);
  Z3_ast invalid_name = z3_api.mk_var("INVALID", pointer_type);
  constraint = Z3_mk_eq(z3_ctx, invalid, invalid_name);
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
  Z3_ast addr_sym = z3_api.mk_var(str.c_str(), addr_space_arr_sort);
  Z3_ast obj_idx = convert_number(idx, config.ansi_c.int_width, true);

  Z3_ast store = Z3_mk_store(z3_ctx, addr_sym, obj_idx, val);

  new_str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num);
  Z3_ast new_addr_sym = z3_api.mk_var(new_str.c_str(),
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

uint
z3_convt::get_z3_core_size(void)
{
  return unsat_core_size;
}

uint
z3_convt::get_z3_number_of_assumptions(void)
{
  return assumptions_status;
}

void
z3_convt::set_z3_core_size(uint val)
{
  if (val)
    max_core_size = val;
}

bool
z3_convt::get_z3_encoding(void) const
{
  return int_encoding;
}

void
z3_convt::set_filename(std::string file)
{
  filename = file;
}

std::string
z3_convt::extract_magnitude(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(0, width / 2), true), 10);
}

std::string
z3_convt::extract_fraction(std::string v, unsigned width)
{
  return integer2string(binary2integer(v.substr(width / 2, width), false), 10);
}

std::string
z3_convt::fixed_point(std::string v, unsigned width)
{
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

void
z3_convt::generate_assumptions(const exprt &expr, const Z3_ast &result)
{
  std::string literal;

  literal = expr.op0().identifier().c_str();
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
    assumpt.push_back(Z3_mk_not(z3_ctx, result));
  } else
    assumpt.push_back(Z3_mk_not(z3_ctx, result));

  // Ensure addrspace array makes its way to the output
  std::string sym = get_cur_addrspace_ident();
  Z3_ast addr_sym = z3_api.mk_var(sym.c_str(), addr_space_arr_sort);
  assumpt.push_back(addr_sym);
}

void
z3_convt::finalize_pointer_chain(void)
{
  bool fixed_model = false;
  unsigned int offs, num_ptrs = addr_space_data.size();
  if (num_ptrs == 0)
    return;

  Z3_sort native_int_sort;
  if (int_encoding)
    native_int_sort = Z3_mk_int_sort(z3_ctx);
  else
    native_int_sort = Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  // Work out which pointer model to take
  if (config.options.get_bool_option("fixed-pointer-model"))
    fixed_model = true;
  else if (config.options.get_bool_option("floating-pointer-model"))
    fixed_model = false;
  // Default - false, non-fixed model.

  if (fixed_model) {
    // Generate fixed model - take each pointer object and ensure that the end
    // of one is immediatetly followed by the start of another. The ordering is
    // in whatever order we originally created the pointer objects. (Or rather,
    // the order that they reached the Z3 backend in).
    offs = 2;
    std::map<unsigned,unsigned>::const_iterator it;
    for (it = addr_space_data.begin(); it != addr_space_data.end(); it++) {

      // The invalid object overlaps everything; it exists to catch anything
      // that slip through the cracks. Don't make the assumption that objects
      // don't overlap it.
      if (it->first == 1)
        continue;

      Z3_ast start = z3_api.mk_var(
                           ("__ESBMC_ptr_obj_start_" + itos(it->first)).c_str(),
                           native_int_sort);
      Z3_ast start_num = convert_number(offs, config.ansi_c.int_width, true);
      Z3_ast eq = Z3_mk_eq(z3_ctx, start, start_num);
      assert_formula(eq);

      offs += it->second - 1;

      Z3_ast end = z3_api.mk_var(
                             ("__ESBMC_ptr_obj_end_" + itos(it->first)).c_str(),
                             native_int_sort);
      Z3_ast end_num = convert_number(offs, config.ansi_c.int_width, true);
      eq = Z3_mk_eq(z3_ctx, end, end_num);
      assert_formula(eq);

      offs++;
    }
  } else {
    // Floating model - we assert that all objects don't overlap each other,
    // but otherwise their locations are entirely defined by Z3. Inefficient,
    // but necessary for accuracy. Unfortunately, has high complexity (O(n^2))

    // Implementation: iterate through all objects; assert that those with lower
    // object nums don't overlap the current one. So for every particular pair
    // of object numbers in the set there'll be a doesn't-overlap clause.

    unsigned num_objs = addr_space_data.size();
    for (unsigned i = 0; i < num_objs; i++) {
      // Obj 1 is designed to overlap
      if (i == 1)
        continue;

       Z3_ast i_start = z3_api.mk_var(
                           ("__ESBMC_ptr_obj_start_" + itos(i)).c_str(),
                           native_int_sort);
      Z3_ast i_end = z3_api.mk_var(
                           ("__ESBMC_ptr_obj_end_" + itos(i)).c_str(),
                           native_int_sort);

      for (unsigned j = 0; j < i; j++) {
        // Obj 1 is designed to overlap
        if (j == 1)
          continue;

        Z3_ast j_start = z3_api.mk_var(
                           ("__ESBMC_ptr_obj_start_" + itos(j)).c_str(),
                           native_int_sort);
        Z3_ast j_end = z3_api.mk_var(
                           ("__ESBMC_ptr_obj_end_" + itos(j)).c_str(),
                           native_int_sort);

        // Formula: (i_end < j_start) || (i_start > j_end)
        // Previous assertions ensure start < end for all objs.
        Z3_ast args[2], formula;
        if (int_encoding) {
          args[0] = Z3_mk_lt(z3_ctx, i_end, j_start);
          args[1] = Z3_mk_gt(z3_ctx, i_start, j_end);
        } else {
          args[0] = Z3_mk_bvult(z3_ctx, i_end, j_start);
          args[1] = Z3_mk_bvugt(z3_ctx, i_start, j_end);
        }
        formula = Z3_mk_or(z3_ctx, 2, args);
        assert_formula(formula);
      }
    }
  }

  return;
}

prop_convt::resultt
z3_convt::dec_solve(void)
{
  unsigned major, minor, build, revision;
  Z3_lbool result;
  Z3_get_version(&major, &minor, &build, &revision);

  // Add assumptions that link up literals to symbols - connections that are
  // made at high level by prop_conv, rather than by the Z3 backend
  link_syms_to_literals();

  std::cout << "Solving with SMT Solver Z3 v" << major << "." << minor << "\n";

  finalize_pointer_chain();

  bv_cache.clear();

  if (smtlib)
    return prop_convt::P_SMTLIB;

  result = check2_z3_properties();

  if (result == Z3_L_FALSE)
    return prop_convt::P_UNSATISFIABLE;
  else if (result == Z3_L_UNDEF)
    return prop_convt::P_ERROR;
  else
    return prop_convt::P_SATISFIABLE;
}

Z3_lbool
z3_convt::check2_z3_properties(void)
{
  Z3_lbool result;
  unsigned i;

  assumptions_status = assumpt.size();

  Z3_ast proof, *core = (Z3_ast *)alloca(assumptions_status * sizeof(Z3_ast)),
         *assumptions_core = (Z3_ast *)alloca(assumptions_status * sizeof(Z3_ast));
  std::string literal;


  if (uw) {
    std::list<Z3_ast>::const_iterator it;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumptions_core[i] = *it;
    }
  }

  try
  {
    if (uw) {
      unsat_core_size = assumpt.size();
      memset(core, 0, sizeof(Z3_ast) * unsat_core_size);
      result = Z3_check_assumptions(z3_ctx, assumpt.size(),
                             assumptions_core, &model, &proof, &unsat_core_size,
                             core);
    } else {
      result = Z3_check_and_get_model(z3_ctx, &model);
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

  if (config.options.get_bool_option("dump-z3-assigns") && model != NULL)
    std::cout << Z3_model_to_string(z3_ctx, model);

  if (uw && result == Z3_L_FALSE)   {
    for (i = 0; i < unsat_core_size; ++i)
    {
      std::string id = Z3_ast_to_string(z3_ctx, core[i]);
      if (id.find("false") != std::string::npos) {
	result = z3_api.check2(Z3_L_TRUE);
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
    Z3_ast sym = z3_api.mk_var(it->first.as_string().c_str(),
                                Z3_mk_bool_sort(z3_ctx));
    Z3_ast formula = Z3_mk_iff(z3_ctx, z3_literal(it->second), sym);
    assert_formula(formula);
 }
}

void
z3_convt::convert_smt_type(const bool_type2t &type __attribute__((unused)),
                           void *&_bv) const
{
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

  bv = Z3_mk_bool_type(z3_ctx);
  return;
}

void
z3_convt::convert_smt_type(const bv_type2t &type, void *&_bv) const
{
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

  if (int_encoding) {
    bv = Z3_mk_int_type(z3_ctx);
  } else {
    unsigned int width = type.get_width();
    bv = Z3_mk_bv_type(z3_ctx, width);
  }

  return;
}

void
z3_convt::convert_smt_type(const array_type2t &type, void *&_bv) const
{
  Z3_sort elem_sort, idx_sort;
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

  if (int_encoding) {
    idx_sort = Z3_mk_int_type(z3_ctx);
  } else {
    idx_sort = Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width);
  }

  type.subtype->convert_smt_type(*this, (void*&)elem_sort);
  bv = Z3_mk_array_type(z3_ctx, idx_sort, elem_sort);

  return;
}

void
z3_convt::convert_smt_type(const pointer_type2t &type __attribute__((unused)),
                           void *&_bv) const
{
  Z3_symbol mk_tuple_name, proj_names[2];
  Z3_type_ast proj_types[2];
  Z3_const_decl_ast mk_tuple_decl, proj_decls[2];
  Z3_sort native_int_sort;
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

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
  return;
}

void
z3_convt::convert_smt_type(const struct_union_type2t &type, void *&_bv) const
{
  Z3_symbol mk_tuple_name, *proj_names;
  std::string name;
  Z3_type_ast *proj_types;
  Z3_const_decl_ast mk_tuple_decl, *proj_decls;
  u_int num_elems;
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

  bool uni = (type.type_id != type2t::struct_id);

  num_elems = type.members.size();
  if (uni)
    num_elems++;

  proj_names = new Z3_symbol[num_elems];
  proj_types = new Z3_type_ast[num_elems];
  proj_decls = new Z3_const_decl_ast[num_elems];

  name = ((uni) ? "union" : "struct" );
  name += "_type_" + type.name;
  mk_tuple_name = Z3_mk_string_symbol(z3_ctx, name.c_str());

  if (!type.members.size()) {
    bv = Z3_mk_tuple_type(z3_ctx, mk_tuple_name, 0, NULL, NULL, &mk_tuple_decl, proj_decls);
    return;
  }

  u_int i = 0;
  std::vector<std::string>::const_iterator mname = type.member_names.begin();
  for (std::vector<type2tc>::const_iterator it = type.members.begin();
       it != type.members.end(); it++, mname++, i++)
  {
    proj_names[i] = Z3_mk_string_symbol(z3_ctx, mname->c_str());
    (*it)->convert_smt_type(*this, (void*&)proj_types[i]);
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

  return;
}

void
z3_convt::convert_smt_type(const fixedbv_type2t &type, void *&_bv) const
{
  Z3_type_ast &bv = (Z3_type_ast &)_bv;

  unsigned int width = type.get_width();

  if (int_encoding)
    bv = Z3_mk_real_type(z3_ctx);
  else
    bv = Z3_mk_bv_type(z3_ctx, width);

  return;
}

void
z3_convt::create_pointer_type(Z3_type_ast &bv) const
{
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

  return;
}

void
z3_convt::convert_smt_expr(const symbol2t &sym, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_sort sort;

  // References to unsigned int identifiers need to be assumed to be > 0,
  // otherwise the solver is free to assign negative nums to it.
  if (is_unsignedbv_type(sym.type) && int_encoding) {
    Z3_ast formula;
    bv = z3_api.mk_int_var(sym.name.c_str());
    formula = Z3_mk_ge(z3_ctx, bv, z3_api.mk_int(0));
    assert_formula(formula);
    return;
  }

  sym.type->convert_smt_type(*this, (void*&)sort);
  bv = z3_api.mk_var(sym.name.c_str(), sort);
}

void
z3_convt::convert_smt_expr(const constant_int2t &sym, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  unsigned int bitwidth = sym.type->get_width();

  Z3_sort int_sort;
  if (int_encoding)
    int_sort = Z3_mk_int_sort(z3_ctx);
  else
    int_sort = Z3_mk_bv_type(z3_ctx, bitwidth);

  if (is_unsignedbv_type(sym.type)) {
    bv = Z3_mk_unsigned_int64(z3_ctx, sym.as_ulong(), int_sort);
  } else {
    assert(is_signedbv_type(sym.type));
    bv = Z3_mk_int64(z3_ctx, sym.as_long(), int_sort);
  }

  return;
}

void
z3_convt::convert_smt_expr(const constant_fixedbv2t &sym, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  unsigned int bitwidth = sym.type->get_width();

  assert(is_fixedbv_type(sym.type));

  std::string theval = sym.value.to_expr().value().as_string();

  if (int_encoding) {
    std::string result = fixed_point(theval, bitwidth);
    bv = Z3_mk_numeral(z3_ctx, result.c_str(), Z3_mk_real_type(z3_ctx));
  } else {
    Z3_ast magnitude, fraction;
    std::string m, f, c;
    m = extract_magnitude(theval, bitwidth);
    f = extract_fraction(theval, bitwidth);
    magnitude =
      Z3_mk_int(z3_ctx, atoi(m.c_str()), Z3_mk_bv_type(z3_ctx, bitwidth / 2));
    fraction =
      Z3_mk_int(z3_ctx, atoi(f.c_str()), Z3_mk_bv_type(z3_ctx, bitwidth / 2));
    bv = Z3_mk_concat(z3_ctx, magnitude, fraction);
  }

  return;
}

void
z3_convt::convert_smt_expr(const constant_bool2t &b, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  if (b.constant_value)
    bv = Z3_mk_true(z3_ctx);
  else
    bv = Z3_mk_false(z3_ctx);
}

void
z3_convt::convert_struct_union(const std::vector<expr2tc> &members,
                               const type2tc &wrappedtype, bool is_union,
                               void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  // Converts a static struct/union - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  const struct_union_type2t &type = to_structure_type(wrappedtype);
  u_int i = 0;

  assert(type.members.size() >= members.size());
  assert(!type.members.empty());

  Z3_sort sort;
  type.convert_smt_type(*this, (void*&)sort);

  unsigned size = type.members.size();
  if (is_union)
    size++;

  Z3_ast *args = (Z3_ast*)alloca(sizeof(Z3_ast) * size);

  unsigned int numoperands = members.size();
  // Populate tuple with members of that struct/union
  forall_exprs(it, members) {
    if (i < numoperands) {
      convert_bv(*it, args[i]);
    } else {
      // Turns out that unions don't necessarily initialize all members.
      // If no initialization give, use free (fresh) variable.
      Z3_sort s;
      (*it)->type->convert_smt_type(*this, (void*&)s);
      args[i] = Z3_mk_fresh_const(z3_ctx, NULL, s);
    }

    i++;
  }

  // Update unions "last-set" member to be the last field
  if (is_union)
    args[size-1] = convert_number(i, config.ansi_c.int_width, false);

  // Create tuple itself, return to caller. This is a lump of data, we don't
  // need to bind it to a name or symbol.
  bv = z3_api.mk_tuple(sort, args, size);
}

void
z3_convt::convert_smt_expr(const constant_struct2t &data, void *&_bv)
{
  convert_struct_union(data.datatype_members, data.type, false, _bv);
}

void
z3_convt::convert_smt_expr(const constant_union2t &data, void *&_bv)
{
  convert_struct_union(data.datatype_members, data.type, true, _bv);
}

void
z3_convt::convert_smt_expr(const constant_array2t &array, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  u_int i = 0;
  Z3_sort native_int_sort;
  Z3_type_ast z3_array_type, elem_type;
  Z3_ast int_cte, val_cte;

  native_int_sort = (int_encoding) ? Z3_mk_int_sort(z3_ctx)
                              : Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  const array_type2t &arr_type = to_array_type(array.type);
  arr_type.subtype->convert_smt_type(*this, (void*&)elem_type);
  z3_array_type = Z3_mk_array_type(z3_ctx, native_int_sort, elem_type);

  bv = Z3_mk_fresh_const(z3_ctx, NULL, z3_array_type);

  i = 0;
  forall_exprs(it, array.datatype_members) {
    int_cte = Z3_mk_int(z3_ctx, i, native_int_sort);

    convert_bv(*it, val_cte);

    bv = Z3_mk_store(z3_ctx, bv, int_cte, val_cte);
    ++i;
  }
}

void
z3_convt::convert_smt_expr(const constant_array_of2t &array, void *&_bv)
{
  Z3_ast value, index;
  Z3_type_ast array_type = 0;
  std::string tmp, identifier;
  int64_t size;
  u_int j;
  Z3_ast &bv = cast_to_z3(_bv);

  const array_type2t &arr = to_array_type(array.type);

  array.type->convert_smt_type(*this, (void*&)array_type);

  if (arr.size_is_infinite) {
    // Don't attempt to do anything with this. The user is on their own.
    bv = Z3_mk_fresh_const(z3_ctx, NULL, array_type);
    return;
  }

  assert(is_constant_int2t(arr.array_size) &&
         "array_of sizes should be constant");

  const constant_int2t &sz = to_constant_int2t(arr.array_size);
  size = sz.as_long();

  convert_bv(array.initializer, value);

  if (is_bool_type(arr.subtype)) {
    value = Z3_mk_false(z3_ctx);
  }

  bv = Z3_mk_fresh_const(z3_ctx, NULL, array_type);

  //update array
  for (j = 0; j < size; j++)
  {
    index = convert_number(j, config.ansi_c.int_width, true);
    bv = Z3_mk_store(z3_ctx, bv, index, value);
  }
}

void
z3_convt::convert_smt_expr(const constant_string2t &str, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  // Convert to array; convert array.
  expr2tc newarray = str.to_array();
  convert_bv(newarray, bv);
  return;
}

void
z3_convt::convert_smt_expr(const if2t &ifirep, void *&_bv)
{
  Z3_ast operand0, operand1, operand2;
  Z3_ast &bv = cast_to_z3(_bv);

  convert_bv(ifirep.cond, operand0);
  convert_bv(ifirep.true_value, operand1);
  convert_bv(ifirep.false_value, operand2);

  bv = Z3_mk_ite(z3_ctx, operand0, operand1, operand2);
  return;
}

void
z3_convt::convert_smt_expr(const equality2t &equality, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(equality.side_1, args[0]);
  convert_bv(equality.side_2, args[1]);

  bv = Z3_mk_eq(z3_ctx, args[0], args[1]);
}

void
z3_convt::convert_smt_expr(const notequal2t &notequal, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(notequal.side_1, args[0]);
  convert_bv(notequal.side_2, args[1]);

  bv = Z3_mk_distinct(z3_ctx, 2, args);
}

void
z3_convt::convert_rel(const expr2tc &side1, const expr2tc &side2,
                      ast_convert_calltype intmode,
                      ast_convert_calltype signedbv,
                      ast_convert_calltype unsignedbv,
                      void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  // XXXjmorse -- pointer comparisons are still broken.
  if (is_pointer_type(side1->type))
    args[0] = z3_api.mk_tuple_select(args[0], 1);

  if (is_pointer_type(side2->type))
    args[1] = z3_api.mk_tuple_select(args[1], 1);

  if (int_encoding) {
    bv = intmode(z3_ctx, args[0], args[1]);
  } else if (is_signedbv_type(side1->type)) {
    bv = signedbv(z3_ctx, args[0], args[1]);
  } else {
    bv = unsignedbv(z3_ctx, args[0], args[1]);
  }
}

void
z3_convt::convert_smt_expr(const lessthan2t &lessthan, void *&_bv)
{
  convert_rel(lessthan.side_1, lessthan.side_2, Z3_mk_lt, Z3_mk_bvslt,
              Z3_mk_bvult, _bv);
}

void
z3_convt::convert_smt_expr(const greaterthan2t &greaterthan, void *&_bv)
{
  convert_rel(greaterthan.side_1, greaterthan.side_2, Z3_mk_gt, Z3_mk_bvsgt,
              Z3_mk_bvugt, _bv);
}

void
z3_convt::convert_smt_expr(const lessthanequal2t &le, void *&_bv)
{
  convert_rel(le.side_1, le.side_2, Z3_mk_le, Z3_mk_bvsle, Z3_mk_bvule, _bv);
}

void
z3_convt::convert_smt_expr(const greaterthanequal2t &ge, void *&_bv)
{
  convert_rel(ge.side_1, ge.side_2, Z3_mk_ge, Z3_mk_bvsge, Z3_mk_bvuge, _bv);
}

void
z3_convt::convert_smt_expr(const not2t &notval, void  *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast z3val;

  convert_bv(notval.value, z3val);
  bv = Z3_mk_not(z3_ctx, z3val);
}

void
z3_convt::convert_logic_2ops(const expr2tc &side1, const expr2tc &side2,
                      ast_convert_calltype converter,
                      ast_convert_multiargs bulkconverter,
                      void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  if (converter != NULL)
    bv = converter(z3_ctx, args[0], args[1]);
  else
    bv = bulkconverter(z3_ctx, 2, args);
}

void
z3_convt::convert_smt_expr(const and2t &andval, void  *&_bv)
{
  convert_logic_2ops(andval.side_1, andval.side_2, NULL, Z3_mk_and, _bv);
}

void
z3_convt::convert_smt_expr(const or2t &orval, void  *&_bv)
{
  convert_logic_2ops(orval.side_1, orval.side_2, NULL, Z3_mk_or, _bv);
}

void
z3_convt::convert_smt_expr(const xor2t &xorval, void  *&_bv)
{
  convert_logic_2ops(xorval.side_1, xorval.side_2, Z3_mk_xor, NULL, _bv);
}

void
z3_convt::convert_smt_expr(const implies2t &implies, void  *&_bv)
{
  convert_logic_2ops(implies.side_1, implies.side_2, Z3_mk_implies, NULL, _bv);
}

void
z3_convt::convert_binop(const expr2tc &side1, const expr2tc &side2,
                        const type2tc &type, ast_convert_calltype converter,
                        void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  // XXXjmorse - int2bv trainwreck.
  if (int_encoding) {
    unsigned int width = side1->type->get_width();
    args[0] = Z3_mk_int2bv(z3_ctx, width, args[0]);
    width = side1->type->get_width();
    args[1] = Z3_mk_int2bv(z3_ctx, width, args[1]);
  }

  bv = converter(z3_ctx, args[0], args[1]);

  if (int_encoding) {
    if (is_signedbv_type(type)) {
      bv = Z3_mk_bv2int(z3_ctx, bv, true);
    } else {
      assert(is_unsignedbv_type(type));
      bv = Z3_mk_bv2int(z3_ctx, bv, false);
    }
  }
}

void
z3_convt::convert_smt_expr(const bitand2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvand, _bv);
}

void
z3_convt::convert_smt_expr(const bitor2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvor, _bv);
}

void
z3_convt::convert_smt_expr(const bitxor2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvxor, _bv);
}

void
z3_convt::convert_smt_expr(const bitnand2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvnand, _bv);
}

void
z3_convt::convert_smt_expr(const bitnor2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvnor, _bv);
}

void
z3_convt::convert_smt_expr(const bitnxor2t &bitval, void *&_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, Z3_mk_bvxnor, _bv);
}

void
z3_convt::convert_smt_expr(const lshr2t &bitval, void *&_bv)
{
  convert_shift(bitval, bitval.side_1, bitval.side_2, Z3_mk_bvlshr, _bv);
}

void
z3_convt::convert_smt_expr(const neg2t &neg, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  convert_bv(neg.value, args[0]);

  if (int_encoding) {
    if (is_bv_type(neg.type)) {
      args[1] = z3_api.mk_int(-1);
    } else {
      assert(is_fixedbv_type(neg.type));
      args[1] = Z3_mk_int(z3_ctx, -1, Z3_mk_real_type(z3_ctx));
    }
    bv = Z3_mk_mul(z3_ctx, 2, args);
  } else   {
    bv = Z3_mk_bvneg(z3_ctx, args[0]);
  }
}

void
z3_convt::convert_smt_expr(const abs2t &abs, void *&_bv)
{
  type2tc sign;
  expr2tc zero;
  Z3_ast &bv = cast_to_z3(_bv);

  if (is_fixedbv_type(abs.type)) {
    sign = abs.type;
    fixedbvt bv; // Defaults to zero.
    bv.spec = fixedbv_spect(64, 32);
    exprt face = bv.to_expr();
    zero = expr2tc(new constant_fixedbv2t(sign, bv));
  } else {
    assert(is_bv_type(abs.type));
    sign = type2tc(new signedbv_type2t(config.ansi_c.int_width));
    zero = expr2tc(new constant_int2t(sign, BigInt(0)));
  }

  expr2tc neg(new neg2t(sign, abs.value));
  expr2tc is_negative(new lessthan2t(abs.value, zero));
  expr2tc result(new if2t(sign, is_negative, neg, abs.value));
  convert_bv(result, bv);
}

void
z3_convt::convert_arith2ops(const expr2tc &side1, const expr2tc &side2,
                            ast_convert_calltype bvconvert,
                            ast_convert_multiargs intmodeconvert,
                            void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast args[2];

  if (is_pointer_type(side1->type) ||
      is_pointer_type(side2->type)) {
    std::cerr << "Pointer arithmetic not implemented for Z3 yet" << std::endl;
    abort();
  }

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  if (int_encoding)
    bv = intmodeconvert(z3_ctx, 2, args);
  else
    bv = bvconvert(z3_ctx, args[0], args[1]);
}

void
z3_convt::convert_smt_expr(const add2t &add, void *&_bv)
{
  if (is_pointer_type(add.type) ||
      is_pointer_type(add.side_1->type) ||
      is_pointer_type(add.side_2->type))
    return convert_pointer_arith(add.expr_id, add.side_1, add.side_2,
                                 add.type, cast_to_z3(_bv));

  convert_arith2ops(add.side_1, add.side_2, Z3_mk_bvadd, Z3_mk_add, _bv);
}

void
z3_convt::convert_smt_expr(const sub2t &sub, void *&_bv)
{
  if (is_pointer_type(sub.type) ||
      is_pointer_type(sub.side_1->type) ||
      is_pointer_type(sub.side_2->type))
    return convert_pointer_arith(sub.expr_id, sub.side_1, sub.side_2,
                                 sub.type,(Z3_ast&)_bv);

  convert_arith2ops(sub.side_1, sub.side_2, Z3_mk_bvsub, Z3_mk_sub, _bv);
}

void
z3_convt::convert_smt_expr(const mul2t &mul, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  if (is_pointer_type(mul.side_1->type) ||
      is_pointer_type(mul.side_2->type)) {
    std::cerr << "Pointer arithmetic not valid in a multiply" << std::endl;
    abort();
  }

  Z3_ast args[2];
  unsigned fraction_bits = 0;

  convert_bv(mul.side_1, args[0]);
  convert_bv(mul.side_2, args[1]);

  if (int_encoding) {
    bv = Z3_mk_mul(z3_ctx, 2, args);
  } else if (!is_fixedbv_type(mul.type)) {
    bv = Z3_mk_bvmul(z3_ctx, args[0], args[1]);
  } else {
    // fixedbv in bv mode. I've no idea if this actually works.
    const fixedbv_type2t &fbvt = to_fixedbv_type(mul.type);
    fraction_bits = fbvt.width - fbvt.integer_bits;
    args[0] = Z3_mk_sign_ext(z3_ctx, fraction_bits, args[0]);
    args[1] = Z3_mk_sign_ext(z3_ctx, fraction_bits, args[1]);
    bv = Z3_mk_bvmul(z3_ctx, args[0], args[1]);
    bv = Z3_mk_extract(z3_ctx, fbvt.width + fraction_bits - 1,
                       fraction_bits, bv);
  }
}

void
z3_convt::convert_smt_expr(const div2t &div, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  assert(!is_pointer_type(div.type) &&
         !is_pointer_type(div.side_1->type) &&
         !is_pointer_type(div.side_2->type) &&
         "Can't divide pointers");

  Z3_ast op0, op1;

  convert_bv(div.side_1, op0);
  convert_bv(div.side_2, op1);

  if (int_encoding) {
    bv = Z3_mk_div(z3_ctx, op0, op1);
  } else   {
    if (is_signedbv_type(div.type)) {
      bv = Z3_mk_bvsdiv(z3_ctx, op0, op1);
    } else if (is_unsignedbv_type(div.type)) {
      bv = Z3_mk_bvudiv(z3_ctx, op0, op1);
    } else {
      // Not the foggiest. Copied from convert_div
      assert(is_fixedbv_type(div.type));
      const fixedbv_type2t &fbvt = to_fixedbv_type(div.type);

      unsigned fraction_bits = fbvt.width - fbvt.integer_bits;

      bv = Z3_mk_extract(z3_ctx, fbvt.width - 1, 0,
                         Z3_mk_bvsdiv(z3_ctx,
                                      Z3_mk_concat(z3_ctx, op0,
                                           convert_number(0, fraction_bits, true)),
                                      Z3_mk_sign_ext(z3_ctx, fraction_bits, op1)));
    }
  }
}

void
z3_convt::convert_smt_expr(const modulus2t &mod, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  assert(!is_pointer_type(mod.type) &&
         !is_pointer_type(mod.side_1->type) &&
         !is_pointer_type(mod.side_2->type) &&
         "Can't modulus pointers");

  Z3_ast op0, op1;

  convert_bv(mod.side_1, op0);
  convert_bv(mod.side_2, op1);

  assert(is_bv_type(mod.type) && "Can only modulus integers");

  if (int_encoding) {
    bv = Z3_mk_mod(z3_ctx, op0, op0);
  } else   {
    if (is_signedbv_type(mod.type)) {
      bv = Z3_mk_bvsrem(z3_ctx, op0, op1);
    } else if (is_unsignedbv_type(mod.type)) {
      bv = Z3_mk_bvurem(z3_ctx, op0, op1);
    }
  }
}

void
z3_convt::convert_shift(const expr2t &shift, const expr2tc &part1,
                        const expr2tc &part2, ast_convert_calltype convert,
                        void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast op0, op1;
  unsigned width_expr, width_op0, width_op1;

  convert_bv(part1, op0);
  convert_bv(part2, op1);

  width_expr = shift.type->get_width();
  width_op0 = part1->type->get_width();
  width_op1 = part2->type->get_width();

  if (int_encoding) {
    op0 = Z3_mk_int2bv(z3_ctx, width_op0, op0);
    op1 = Z3_mk_int2bv(z3_ctx, width_op1, op1);
  }

  if (width_op0 > width_expr)
    op0 = Z3_mk_extract(z3_ctx, (width_expr - 1), 0, op0);
  if (width_op1 > width_expr)
    op1 = Z3_mk_extract(z3_ctx, (width_expr - 1), 0, op1);

  if (width_op0 > width_op1) {
    if (is_unsignedbv_type(part1->type))
      op1 = Z3_mk_zero_ext(z3_ctx, (width_op0 - width_op1), op1);
    else
      op1 = Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op1), op1);
  }

  bv = convert(z3_ctx, op0, op1);

  if (int_encoding) {
    if (is_signedbv_type(shift.type)) {
      bv = Z3_mk_bv2int(z3_ctx, bv, true);
    } else {
      assert(is_unsignedbv_type(shift.type));
      bv = Z3_mk_bv2int(z3_ctx, bv, false);
    }
  }
}

void
z3_convt::convert_smt_expr(const shl2t &shl, void *&_bv)
{
  convert_shift(shl, shl.side_1, shl.side_2, Z3_mk_bvshl, _bv);
}

void
z3_convt::convert_smt_expr(const ashr2t &ashr, void *&_bv)
{
  convert_shift(ashr, ashr.side_1, ashr.side_2, Z3_mk_bvashr, _bv);
}

void
z3_convt::convert_smt_expr(const same_object2t &same, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast pointer[2], objs[2];

  assert(is_pointer_type(same.side_1->type));
  assert(is_pointer_type(same.side_2->type));

  convert_bv(same.side_1, pointer[0]);
  convert_bv(same.side_2, pointer[1]);

  objs[0] = z3_api.mk_tuple_select(pointer[0], 0);
  objs[1] = z3_api.mk_tuple_select(pointer[1], 0);
  bv = Z3_mk_eq(z3_ctx, objs[0], objs[1]);
}

void
z3_convt::convert_smt_expr(const pointer_offset2t &offs, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast pointer;

  convert_bv(offs.ptr_obj, pointer);

  bv = z3_api.mk_tuple_select(pointer, 1); //select pointer offset
}

void
z3_convt::convert_smt_expr(const pointer_object2t &obj, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast pointer;

  convert_bv(obj.ptr_obj, pointer);

  bv = z3_api.mk_tuple_select(pointer, 0); //select pointer offset
}

void
z3_convt::convert_smt_expr(const address_of2t &obj, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_type_ast pointer_type;
  std::string symbol_name, out;

  obj.type->convert_smt_type(*this, (void*&)pointer_type);

  if (is_index2t(obj.ptr_obj)) {
    const index2t &idx = to_index2t(obj.ptr_obj);

    if (!is_string_type(idx.source_value->type)) {
      const array_type2t &arr = to_array_type(idx.source_value->type);

      // Pick pointer-to array subtype; need to make pointer arith work.
      expr2tc addrof(new address_of2t(arr.subtype, idx.source_value));
      expr2tc plus(new add2t(addrof->type, addrof, idx.index));
      convert_bv(plus, bv);
    } else {
      // Strings; convert with slightly different types.
      type2tc stringtype(new unsignedbv_type2t(8));
      expr2tc addrof(new address_of2t(stringtype, idx.source_value));
      expr2tc plus(new add2t(addrof->type, addrof, idx.index));
      convert_bv(plus, bv);
    }
  } else if (is_member2t(obj.ptr_obj)) {
    const member2t &memb = to_member2t(obj.ptr_obj);

    int64_t offs;
    if (is_struct_type(memb.source_value->type)) {
      const struct_type2t &type = to_struct_type(memb.source_value->type);
      offs = member_offset(type, memb.member).to_long();
    } else {
      offs = 0; // Offset is always zero for unions.
    }

    expr2tc addr(new address_of2t(type2tc(
                                   new pointer_type2t(memb.source_value->type)),
                       memb.source_value));

    convert_bv(addr, bv);

    // Update pointer offset to offset to that field.
    Z3_ast num = convert_number(offs, config.ansi_c.int_width, true);
    bv = z3_api.mk_tuple_update(bv, 1, num);
  } else if (is_symbol2t(obj.ptr_obj)) {
// XXXjmorse             obj.ptr_obj->expr_id == expr2t::code_id) {

    const symbol2t &symbol = to_symbol2t(obj.ptr_obj);
    convert_identifier_pointer(obj.ptr_obj, symbol.name.as_string(), bv);
  } else if (is_constant_string2t(obj.ptr_obj)) {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    const constant_string2t &str = to_constant_string2t(obj.ptr_obj);
    std::string identifier = "address_of_str_const(" + str.value + ")";
    convert_identifier_pointer(obj.ptr_obj, identifier, bv);
  } else if (is_if2t(obj.ptr_obj)) {
    // We can't nondeterministically take the address of something; So instead
    // rewrite this to be if (cond) ? &a : &b;.

    const if2t &ifval = to_if2t(obj.ptr_obj);

    expr2tc addrof1(new address_of2t(obj.type, ifval.true_value));
    expr2tc addrof2(new address_of2t(obj.type, ifval.false_value));
    expr2tc newif(new if2t (obj.type, ifval.cond, addrof1, addrof2));
    convert_bv(newif, bv);
  } else {
    throw new conv_error("Unrecognized address_of operand");
  }
}



void
z3_convt::convert_smt_expr(const byte_extract2t &data, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  if (!is_constant_int2t(data.source_offset))
    throw new conv_error("byte_extract expects constant 2nd arg");

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  unsigned width;
  width = data.source_value->type->get_width();
  // XXXjmorse - looks like this only ever reads a single byte, not the desired
  // number of bytes to fill the type.

  uint64_t upper, lower;
  if (!data.big_endian) {
    upper = ((intref.constant_value.to_long() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.constant_value.to_long() * 8; //i*w;
  } else {
    uint64_t max = width - 1;
    upper = max - (intref.constant_value.to_long() * 8); //max-(i*w);
    lower = max - ((intref.constant_value.to_long() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  Z3_ast source;

  convert_bv(data.source_value, source);

  if (int_encoding) {
    if (is_fixedbv_type(data.source_value->type)) {
      if (is_bv_type(data.type)) {
	Z3_ast tmp;
	source = Z3_mk_real2int(z3_ctx, source);
	tmp = Z3_mk_int2bv(z3_ctx, width, source);
	bv = Z3_mk_extract(z3_ctx, upper, lower, tmp);
	if (is_signedbv_type(data.type))
	  bv = Z3_mk_bv2int(z3_ctx, bv, 1);
	else
	  bv = Z3_mk_bv2int(z3_ctx, bv, 0);
      } else {
	throw new conv_error("unsupported type for byte_extract");
      }
    } else if (is_bv_type(data.source_value->type)) {
      Z3_ast tmp;
      tmp = Z3_mk_int2bv(z3_ctx, width, source);

      if (width >= upper)
	bv = Z3_mk_extract(z3_ctx, upper, lower, tmp);
      else
	bv = Z3_mk_extract(z3_ctx, upper - lower, 0, tmp);

      if (is_signedbv_type(data.source_value->type))
	bv = Z3_mk_bv2int(z3_ctx, bv, 1);
      else
	bv = Z3_mk_bv2int(z3_ctx, bv, 0);
    } else {
      throw new conv_error("unsupported type for byte_extract");
    }
  } else {
    if (is_struct_type(data.source_value->type)) {
      const struct_type2t &struct_type =to_struct_type(data.source_value->type);
      unsigned i = 0, num_elems = struct_type.members.size();
      Z3_ast struct_elem[num_elems + 1], struct_elem_inv[num_elems + 1];

      forall_types(it, struct_type.members) {
        struct_elem[i] = z3_api.mk_tuple_select(source, i);
        i++;
      }

      for (unsigned k = 0; k < num_elems; k++)
        struct_elem_inv[(num_elems - 1) - k] = struct_elem[k];

      for (unsigned k = 0; k < num_elems; k++)
      {
        if (k == 1)
          struct_elem_inv[num_elems] = Z3_mk_concat(
            z3_ctx, struct_elem_inv[k - 1], struct_elem_inv[k]);
        else if (k > 1)
          struct_elem_inv[num_elems] = Z3_mk_concat(
            z3_ctx, struct_elem_inv[num_elems], struct_elem_inv[k]);
      }

      source = struct_elem_inv[num_elems];
    }

    bv = Z3_mk_extract(z3_ctx, upper, lower, source);
  }
}

void
z3_convt::convert_smt_expr(const byte_update2t &data, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  if (!is_constant_int2t(data.source_offset))
    throw new conv_error("byte_extract expects constant 2nd arg");

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  Z3_ast tuple, value;
  uint width_op0, width_op2;

  convert_bv(data.source_value, tuple);
  convert_bv(data.update_value, value);

  width_op2 = data.update_value->type->get_width();

  if (is_struct_type(data.source_value->type)) {
    const struct_type2t &struct_type = to_struct_type(data.source_value->type);
    bool has_field = false;

    // XXXjmorse, this isn't going to be the case if it's a with.

    forall_types(it, struct_type.members) {
      width_op0 = (*it)->get_width();

      if (((*it)->type_id == data.update_value->type->type_id) &&
          (width_op0 == width_op2))
	has_field = true;
    }

    if (has_field)
      bv = z3_api.mk_tuple_update(tuple, intref.constant_value.to_long(), value);
    else
      bv = tuple;
  } else if (is_signedbv_type(data.source_value->type)) {
    if (int_encoding) {
      bv = value;
      return;
    }

    width_op0 = data.source_value->type->get_width();

    if (width_op0 == 0)
      // XXXjmorse - can this ever happen now?
      throw new conv_error("failed to get width of byte_update operand");

    if (width_op0 > width_op2)
      bv = Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op2), value);
    else
      throw new conv_error("unsupported irep for conver_byte_update");
  } else {
    throw new conv_error("unsupported irep for conver_byte_update");
  }

}

void
z3_convt::convert_smt_expr(const with2t &with, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast operand0, operand1, operand2;
  Z3_ast tuple, value;

  if (is_structure_type(with.type)) {
    unsigned int idx = 0;
    const struct_union_type2t &struct_type = to_structure_type(with.type);

    convert_bv(with.source_value, tuple);
    convert_bv(with.update_value, value);

    const constant_string2t &str = to_constant_string2t(with.update_field);

    forall_names(it, struct_type.member_names) {
      if (*it == str.value)
        break;
      idx++;
    }

    assert(idx != struct_type.member_names.size() &&
           "Member name of with expr not found in struct/union type");

    bv = z3_api.mk_tuple_update(tuple, idx, value);

    // Update last-updated-field field if it's a union
    if (is_union_type(with.type)) {
      const union_type2t &unionref = to_union_type(with.type);
       unsigned int components_size = unionref.members.size();
       bv = z3_api.mk_tuple_update(bv, components_size,
                              convert_number(idx, config.ansi_c.int_width, 0));
    }
  } else if (is_array_type(with.type)) {

    convert_bv(with.source_value, operand0);
    convert_bv(with.update_field, operand1);
    convert_bv(with.update_value, operand2);

    bv = Z3_mk_store(z3_ctx, operand0, operand1, operand2);
  } else {
    throw new conv_error("with applied to non-struct/union/array obj");
  }
}

void
z3_convt::convert_smt_expr(const member2t &member, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  u_int j = 0;
  Z3_ast struct_var;

  const struct_union_type2t &struct_type =
    to_structure_type(member.source_value->type);

  forall_names(it, struct_type.member_names) {
    if (*it == member.member.as_string())
      break;
    j++;
  }

  convert_bv(member.source_value, struct_var);

  if (is_union_type(member.source_value->type)) {
    union_varst::const_iterator cache_result;

    if (is_symbol2t(member.source_value)) {
      const symbol2t &sym = to_symbol2t(member.source_value);
      cache_result = union_vars.find(sym.name.as_string().c_str());
    } else {
      cache_result = union_vars.end();
    }

    if (cache_result != union_vars.end()) {
      const struct_union_type2t &type =
        to_structure_type(member.source_value->type);

      const type2tc source_type = type.members[cache_result->second];
      if (source_type == member.type) {
        // Type we're fetching from union matches expected type; just return it.
        bv = z3_api.mk_tuple_select(struct_var, cache_result->second);
        return;
      }

      // Union field and expected type mismatch. Need to insert a cast.
      // Duplicate expr as we're changing it
      expr2tc memb2(new member2t(source_type, member.source_value, member.member));
      expr2tc cast(new typecast2t(member.type, memb2));
      convert_bv(cast, bv);
      return;
    }
  }

  bv = z3_api.mk_tuple_select(struct_var, j);
}

void
z3_convt::convert_typecast_bool(const typecast2t &cast, Z3_ast &bv)
{
  Z3_ast args[2];

  if (is_bv_type(cast.from->type) ||
      is_pointer_type(cast.from->type)) {
    args[0] = bv;
    if (int_encoding)
      args[1] = z3_api.mk_int(0);
    else
      args[1] =
        Z3_mk_int(z3_ctx, 0, Z3_mk_bv_type(z3_ctx, config.ansi_c.int_width));

    bv = Z3_mk_distinct(z3_ctx, 2, args);
  } else {
    throw new conv_error("Unimplemented bool typecast");
  }
}

void
z3_convt::convert_typecast_fixedbv_nonint(const typecast2t &cast, Z3_ast &bv)
{

  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;

  if (is_pointer_type(cast.from->type)) {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  if (is_bv_type(cast.from->type)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_integer_bits) {
      ; // No-op, already converted by higher caller
    } else if (from_width > to_integer_bits) {
      bv = Z3_mk_extract(z3_ctx, (from_width - 1), to_integer_bits, bv);
    } else {
      assert(from_width < to_integer_bits);
      bv = Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_width), bv);
    }

    bv = Z3_mk_concat(z3_ctx, bv, convert_number(0, to_fraction_bits, true));
  } else if (is_bool_type(cast.from->type)) {
    Z3_ast zero, one;
    zero = convert_number(0, to_integer_bits, true);
    one =  convert_number(1, to_integer_bits, true);
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
    bv = Z3_mk_concat(z3_ctx, bv, convert_number(0, to_fraction_bits, true));
  } else if (is_fixedbv_type(cast.from->type)) {
    Z3_ast magnitude, fraction;

    const fixedbv_type2t &from_fbvt = to_fixedbv_type(cast.from->type);

    unsigned from_fraction_bits = from_fbvt.width - from_fbvt.integer_bits;
    unsigned from_integer_bits = from_fbvt.integer_bits;
    unsigned from_width = from_fbvt.width;

    if (to_integer_bits <= from_integer_bits) {
      magnitude =
        Z3_mk_extract(z3_ctx, (from_fraction_bits + to_integer_bits - 1),
                      from_fraction_bits, bv);
    } else   {
      assert(to_integer_bits > from_integer_bits);

      magnitude =
        Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_integer_bits),
                       Z3_mk_extract(z3_ctx, from_width - 1, from_fraction_bits,
                                     bv));
    }

    if (to_fraction_bits <= from_fraction_bits) {
      fraction =
        Z3_mk_extract(z3_ctx, (from_fraction_bits - 1),
                      from_fraction_bits - to_fraction_bits,
                      bv);
    } else   {
      assert(to_fraction_bits > from_fraction_bits);
      fraction =
        Z3_mk_concat(z3_ctx,
                     Z3_mk_extract(z3_ctx, (from_fraction_bits - 1), 0, bv),
                     convert_number(0, to_fraction_bits - from_fraction_bits,
                                    true));
    }
    bv = Z3_mk_concat(z3_ctx, magnitude, fraction);
  } else {
    throw new conv_error("unexpected typecast to fixedbv");
  }

  return;
}

void
z3_convt::convert_typecast_to_ints(const typecast2t &cast, Z3_ast &bv)
{
  unsigned to_width = cast.type->get_width();

  if (is_signedbv_type(cast.from->type) ||
      is_fixedbv_type(cast.from->type)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      if (int_encoding && is_signedbv_type(cast.from->type) &&
               is_fixedbv_type(cast.type))
	bv = Z3_mk_int2real(z3_ctx, bv);
      else if (int_encoding && is_fixedbv_type(cast.from->type) &&
               is_signedbv_type(cast.type))
	bv = Z3_mk_real2int(z3_ctx, bv);
      // XXXjmorse - there isn't a case here for if !int_encoding

    } else if (from_width < to_width)      {
      if (int_encoding &&
          ((is_fixedbv_type(cast.type) &&
            is_signedbv_type(cast.from->type))))
	bv = Z3_mk_int2real(z3_ctx, bv);
      else if (int_encoding)
	; // bv = bv
      else
	bv = Z3_mk_sign_ext(z3_ctx, (to_width - from_width), bv);
    } else if (from_width > to_width)     {
      if (int_encoding &&
          ((is_signedbv_type(cast.from->type) &&
            is_fixedbv_type(cast.type))))
	bv = Z3_mk_int2real(z3_ctx, bv);
      else if (int_encoding &&
               (is_fixedbv_type(cast.from->type) &&
                is_signedbv_type(cast.type)))
	bv = Z3_mk_real2int(z3_ctx, bv);
      else if (int_encoding)
	; // bv = bv
      else {
	if (!to_width) to_width = config.ansi_c.int_width;
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, bv);
      }
    }
  } else if (is_unsignedbv_type(cast.from->type)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      ; // bv = bv
    } else if (from_width < to_width)      {
      if (int_encoding)
	; // bv = bv
      else
	bv = Z3_mk_zero_ext(z3_ctx, (to_width - from_width), bv);
    } else if (from_width > to_width)     {
      if (int_encoding)
	; // bv = bv
      else
	bv = Z3_mk_extract(z3_ctx, (to_width - 1), 0, bv);
    }
  } else if (is_bool_type(cast.from->type)) {
    Z3_ast zero = 0, one = 0;
    unsigned width = cast.type->get_width();

    if (is_signedbv_type(cast.type)) {
      if (int_encoding) {
	zero = Z3_mk_int(z3_ctx, 0, Z3_mk_int_type(z3_ctx));
	one = Z3_mk_int(z3_ctx, 1, Z3_mk_int_type(z3_ctx));
      } else   {
	zero = convert_number(0, width, true);
	one =  convert_number(1, width, true);
      }
    } else if (is_unsignedbv_type(cast.type)) {
      if (int_encoding) {
	zero = Z3_mk_int(z3_ctx, 0, Z3_mk_int_type(z3_ctx));
	one = Z3_mk_int(z3_ctx, 1, Z3_mk_int_type(z3_ctx));
      } else   {
	zero = convert_number(0, width, false);
	one =  convert_number(1, width, false);
      }
    } else if (is_fixedbv_type(cast.type)) {
      zero = Z3_mk_numeral(z3_ctx, "0", Z3_mk_real_type(z3_ctx));
      one = Z3_mk_numeral(z3_ctx, "1", Z3_mk_real_type(z3_ctx));
    } else {
      throw new conv_error("Unexpected type in typecast of bool");
    }
    bv = Z3_mk_ite(z3_ctx, bv, one, zero);
  } else   {
    throw new conv_error("Unexpected type in int/ptr typecast");
  }
}

void
z3_convt::convert_typecast_struct(const typecast2t &cast, Z3_ast &bv)
{
  const struct_type2t &struct_type_from = to_struct_type(cast.from->type);
  const struct_type2t &struct_type_to = to_struct_type(cast.type);

  Z3_ast freshval;
  u_int i = 0, i2 = 0;

  std::vector<type2tc> new_members;
  std::vector<std::string> new_names;
  new_members.reserve(struct_type_to.members.size());
  new_names.reserve(struct_type_to.members.size());

  forall_types(it2, struct_type_to.members) {
    i = 0;
    forall_types(it, struct_type_from.members) {
      if (struct_type_from.member_names[i] == struct_type_to.member_names[i2]) {
	unsigned width = (*it)->get_width();

	if (is_signedbv_type(*it)) {
          new_members.push_back(type2tc(new signedbv_type2t(width)));
	} else if (is_unsignedbv_type(*it)) {
          new_members.push_back(type2tc(new unsignedbv_type2t(width)));
	} else if (is_bool_type(*it))     {
          new_members.push_back(type2tc(new bool_type2t()));
	} else {
          throw new conv_error("Unexpected type when casting struct");
	}
        new_names.push_back(struct_type_from.member_names[i]);
      }

      i++;
    }

    i2++;
  }

  struct_type2t newstruct(new_members, new_names, struct_type_to.name);
  Z3_sort sort;
  newstruct.convert_smt_type(*this, (void*&)sort);

  freshval = Z3_mk_fresh_const(z3_ctx, NULL, sort);

  i2 = 0;
  forall_types(it, newstruct.members) {
    Z3_ast formula;
    formula = Z3_mk_eq(z3_ctx, z3_api.mk_tuple_select(freshval, i2),
                       z3_api.mk_tuple_select(bv, i2));
    assert_formula(formula);
    i2++;
  }

  bv = freshval;
  return;
}

void
z3_convt::convert_typecast_to_ptr(const typecast2t &cast, Z3_ast &bv)
{

  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if (is_pointer_type(cast.from->type)) {
    // bv is already plain-converted.
    return;
  }

  // Unpleasentness; we don't know what pointer this integer is going to
  // correspond to, and there's no way of telling statically, so we have
  // to enumerate all pointers it could point at. IE, all of them. Which
  // is expensive, but here we are.

  // First cast it to an unsignedbv
  Z3_ast target;
  type2tc int_type(new unsignedbv_type2t(config.ansi_c.int_width));
  expr2tc cast_to_unsigned(new typecast2t(int_type, cast.from));
  convert_bv(cast_to_unsigned, target);

  // Construct array for all possible object outcomes
  Z3_ast *is_in_range = (Z3_ast*)alloca(sizeof(Z3_ast) * addr_space_data.size());
  Z3_ast *obj_ids = (Z3_ast*)alloca(sizeof(Z3_ast) * addr_space_data.size());
  Z3_ast *obj_starts = (Z3_ast*)alloca(sizeof(Z3_ast) * addr_space_data.size());

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
    Z3_ast start = z3_api.mk_var(
                                 ("__ESBMC_ptr_obj_start_" + itos(id)).c_str(),
                                 native_int_sort);
    Z3_ast end = z3_api.mk_var(
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
z3_convt::convert_typecast_from_ptr(const typecast2t &cast, Z3_ast &bv)
{
  type2tc int_type(new unsignedbv_type2t(config.ansi_c.int_width));

  // The plan: index the object id -> address-space array and pick out the
  // start address, then add it to any additional pointer offset.

  // Generate type of address space array
  std::vector<type2tc> members;
  std::vector<std::string> names;
  type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
  members.push_back(inttype);
  members.push_back(inttype);
  names.push_back("start");
  names.push_back("end");
  type2tc strct(new struct_type2t(members, names, "addr_space_tuple"));
  type2tc addrspace_type(new array_type2t(strct, expr2tc((expr2t*)NULL), true));

  expr2tc obj_num(new pointer_object2t(inttype, cast.from));

  expr2tc addrspacesym(new symbol2t(addrspace_type, get_cur_addrspace_ident()));
  expr2tc idx(new index2t(strct, addrspacesym, obj_num));

  // We've now grabbed the pointer struct, now get first element
  expr2tc memb(new member2t(int_type, idx, irep_idt("start")));

  expr2tc ptr_offs(new pointer_offset2t(int_type, cast.from));
  expr2tc add(new add2t(int_type, memb, ptr_offs));

  // Finally, replace typecast
  expr2tc new_cast(new typecast2t(cast.type, add));
  convert_bv(new_cast, bv);
}

void
z3_convt::convert_smt_expr(const typecast2t &cast, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  convert_bv(cast.from, bv);

  if (is_pointer_type(cast.type)) {
    convert_typecast_to_ptr(cast, bv);
  } else if (is_pointer_type(cast.from->type)) {
    convert_typecast_from_ptr(cast, bv);
  } else if (is_bool_type(cast.type)) {
    convert_typecast_bool(cast, bv);
  } else if (is_fixedbv_type(cast.type) && !int_encoding)      {
    convert_typecast_fixedbv_nonint(cast, bv);
  } else if (is_bv_type(cast.type) ||
             is_fixedbv_type(cast.type) ||
             is_pointer_type(cast.type)) {
    convert_typecast_to_ints(cast, bv);
  } else if (is_struct_type(cast.type))     {
    convert_typecast_struct(cast, bv);
  } else {
    // XXXjmorse -- what about all other types, eh?
    throw new conv_error("Typecast for unexpected type");
  }
}

void
z3_convt::convert_smt_expr(const index2t &index, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  Z3_ast source, idx;

  convert_bv(index.source_value, source);
  convert_bv(index.index, idx);

  // XXXjmorse - consider situation where a pointer is indexed. Should it
  // give the address of ptroffset + (typesize * index)?
  bv = Z3_mk_select(z3_ctx, source, idx);
}

void
z3_convt::convert_smt_expr(const zero_string2t &zstr, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  // XXXjmorse - this method appears to just return a free variable. Surely
  // it should be selecting the zero_string field out of the referenced
  // string?
  Z3_type_ast array_type;

  zstr.type->convert_smt_type(*this, (void*&)array_type);

  bv = z3_api.mk_var("zero_string", array_type);
}

void
z3_convt::convert_smt_expr(const zero_length_string2t &s, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast operand;

  convert_bv(s.string, operand);
  bv = z3_api.mk_tuple_select(operand, 0);
}

void
z3_convt::convert_smt_expr(const isnan2t &isnan, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);

  if (is_fixedbv_type(isnan.value->type)) {
    Z3_ast op0;
    unsigned width = isnan.value->type->get_width();

    convert_bv(isnan.value, op0);

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
    throw new conv_error("isnan with unsupported operand type");
  }
}

void
z3_convt::convert_smt_expr(const overflow2t &overflow, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast result[2], operand[2];
  unsigned width_op0, width_op1;

  // XXX jmorse - we can't tell whether or not we're supposed to be treating
  // the _result_ as being a signedbv or an unsignedbv, because we only have
  // operands. Ideally, this needs to be encoded somewhere.
  // Specifically, when irep2 conversion reaches code creation, we should
  // encode the resulting type in the overflow operands type. Right now it's
  // inferred.
  bool is_signed = false;

  typedef Z3_ast (*type1)(Z3_context, Z3_ast, Z3_ast, Z3_bool);
  typedef Z3_ast (*type2)(Z3_context, Z3_ast, Z3_ast);
  type1 call1;
  type2 call2;

  // Unseen downside of flattening templates. Should consider reformatting
  // typecast2t.
  if (is_add2t(overflow.operand)) {
    convert_bv(to_add2t(overflow.operand).side_1, operand[0]);
    convert_bv(to_add2t(overflow.operand).side_2, operand[1]);
    width_op0 = to_add2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_add2t(overflow.operand).side_2->type->get_width();
    call1 = Z3_mk_bvadd_no_overflow;
    call2 = Z3_mk_bvadd_no_underflow;
    if (is_signedbv_type(to_add2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_add2t(overflow.operand).side_2->type))
    is_signed = true;
  } else if (is_sub2t(overflow.operand)) {
    convert_bv(to_sub2t(overflow.operand).side_1, operand[0]);
    convert_bv(to_sub2t(overflow.operand).side_2, operand[1]);
    width_op0 = to_sub2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_sub2t(overflow.operand).side_2->type->get_width();
    call1 = Z3_mk_bvsub_no_underflow;
    call2 = Z3_mk_bvsub_no_overflow;
    if (is_signedbv_type(to_sub2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_sub2t(overflow.operand).side_2->type))
    is_signed = true;
  } else if (is_mul2t(overflow.operand)) {
    convert_bv(to_mul2t(overflow.operand).side_1, operand[0]);
    convert_bv(to_mul2t(overflow.operand).side_2, operand[1]);
    width_op0 = to_mul2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_mul2t(overflow.operand).side_2->type->get_width();
    call1 = Z3_mk_bvmul_no_overflow;
    call2 = Z3_mk_bvmul_no_underflow;
    if (is_signedbv_type(to_mul2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_mul2t(overflow.operand).side_2->type))
    is_signed = true;
  } else {
    std::cerr << "Overflow operation with invalid operand";
    abort();
  }

  // XXX jmorse - int2bv trainwreck.
  if (int_encoding) {
    operand[0] = Z3_mk_int2bv(z3_ctx, width_op0, operand[0]);
    operand[1] = Z3_mk_int2bv(z3_ctx, width_op1, operand[1]);
  }

  result[0] = call1(z3_ctx, operand[0], operand[1], is_signed);
  result[1] = call2(z3_ctx, operand[0], operand[1]);
  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, result));
}

void
z3_convt::convert_smt_expr(const overflow_cast2t &ocast, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast operand[3], mid, overflow[2], tmp, minus_one, two;
  uint64_t result;
  u_int width;

  width = ocast.operand->type->get_width();

  if (ocast.bits >= width || ocast.bits == 0)
    throw new conv_error("overflow-typecast got wrong number of bits");

  assert(ocast.bits <= 32 && ocast.bits != 0);
  result = 1 << ocast.bits;

  convert_bv(ocast.operand, operand[0]);

  // XXXjmorse - int2bv trainwreck.
  if (int_encoding)
    operand[0] = Z3_mk_int2bv(z3_ctx, width, operand[0]);

  // XXXjmorse - fixedbv is /not/ always partitioned at width/2
  if (is_fixedbv_type(ocast.operand->type)) {
    unsigned size = (width / 2) + 1;
    operand[0] = Z3_mk_extract(z3_ctx, width - 1, size - 1, operand[0]);
  }

  if (is_signedbv_type(ocast.operand->type) ||
      is_fixedbv_type(ocast.operand->type)) {
    // Produce some useful constants
    unsigned int nums_width = (is_signedbv_type(ocast.operand->type))
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
  } else if (is_unsignedbv_type(ocast.operand->type)) {
    // Create zero and 2^bitwidth,
    operand[2] = convert_number_bv(0, width, false);
    operand[1] = convert_number_bv(result, width, false);
    // Ensure operand lies between those numbers.
    overflow[0] = Z3_mk_bvult(z3_ctx, operand[0], operand[1]);
    overflow[1] = Z3_mk_bvuge(z3_ctx, operand[0], operand[2]);
  }

  bv = Z3_mk_not(z3_ctx, Z3_mk_and(z3_ctx, 2, overflow));
}

void
z3_convt::convert_smt_expr(const overflow_neg2t &neg, void *&_bv)
{
  Z3_ast &bv = cast_to_z3(_bv);
  Z3_ast operand;
  unsigned width;

  convert_bv(neg.operand, operand);

  // XXX jmorse - clearly wrong. Neg of pointer?
  if (is_pointer_type(neg.operand->type))
    operand = z3_api.mk_tuple_select(operand, 1);

  width = neg.operand->type->get_width();

  // XXX jmorse - int2bv trainwreck
  if (int_encoding)
    operand = Z3_mk_int2bv(z3_ctx, width, operand);

  bv = Z3_mk_not(z3_ctx, Z3_mk_bvneg_no_overflow(z3_ctx, operand));
}

void
z3_convt::convert_pointer_arith(expr2t::expr_ids id, const expr2tc &side1,
                                const expr2tc &side2,
                                const type2tc &type, Z3_ast &bv)
{

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
  ret_is_ptr = (is_pointer_type(type)) ? 4 : 0;
  op1_is_ptr = (is_pointer_type(side1->type)) ? 2 : 0;
  op2_is_ptr = (is_pointer_type(side2->type)) ? 1 : 0;

  switch (ret_is_ptr | op1_is_ptr | op2_is_ptr) {
    case 0:
      assert(false);
      break;
    case 3:
    case 7:
      throw new conv_error("Pointer arithmatic with two pointer operands");
      break;
    case 4:
      // Artithmatic operation that has the result type of ptr.
      // Should have been handled at a higher level
      throw new conv_error("Non-pointer op being interpreted as pointer without"
                           " typecast");
      break;
    case 1:
    case 2:
      { // Block required to give a variable lifetime to the cast/add variables
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      expr2tc add(new add2t(ptr_op->type, ptr_op, non_ptr_op));
      // That'll generate the correct pointer arithmatic; now typecast
      expr2tc cast(new typecast2t(type, add));
      convert_bv(cast, bv);
      break;
      }
    case 5:
    case 6:
      {
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      // Actually perform some pointer arith
      const pointer_type2t &ptr_type = to_pointer_type(ptr_op->type);
      mp_integer type_size = pointer_offset_size(*ptr_type.subtype.get());

      // Generate nonptr * constant.
      type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
      expr2tc constant(new constant_int2t(inttype, type_size));
      expr2tc mul(new mul2t(inttype, non_ptr_op, constant));

      // Add or sub that value
      expr2tc ptr_offset(new pointer_offset2t(inttype, ptr_op));

      expr2tc newexpr;
      if (id == expr2t::add_id) {
        newexpr = expr2tc(new add2t(inttype, mul, ptr_offset));
      } else {
        // Preserve order for subtraction.
        expr2tc tmp_op1 = (op1_is_ptr) ? ptr_offset : mul;
        expr2tc tmp_op2 = (op1_is_ptr) ? mul : ptr_offset;
        newexpr = expr2tc(new sub2t(inttype, tmp_op1, tmp_op2));
      }

      // Voila, we have our pointer arithmatic
      convert_bv(newexpr, bv);

      // That calculated the offset; update field in pointer.
      Z3_ast the_ptr;
      convert_bv(ptr_op, the_ptr);
      bv = z3_api.mk_tuple_update(the_ptr, 1, bv);

      break;
      }
  }
}

void
z3_convt::convert_bv(const expr2tc &expr, Z3_ast &bv)
{

  bv_cachet::const_iterator cache_result = bv_cache.find(expr);
  if (cache_result != bv_cache.end()) {
    bv = cache_result->second;
    return;
  }

  expr->convert_smt(*this, (void *&)bv);

  // insert into cache
  bv_cache.insert(std::pair<const expr2tc, Z3_ast>(expr, bv));
  return;
}

void
z3_convt::convert_type(const type2tc &type, Z3_sort &outtype)
{

  sort_cachet::const_iterator cache_result = sort_cache.find(type);
  if (cache_result != sort_cache.end()) {
    outtype = cache_result->second;
    return;
  }

  type->convert_smt_type(*this, (void *&)outtype);

  // insert into cache
  sort_cache.insert(std::pair<const type2tc, Z3_sort>(type, outtype));
  return;
}

literalt
z3_convt::convert_expr(const expr2tc &expr)
{
  literalt l = new_variable();
  Z3_ast formula, constraint;

  expr2tc new_expr;

  try {
    expr->convert_smt(*this, (void *&)constraint);
  } catch (std::string *e) {
    std::cerr << "Failed to convert an expression" << std::endl;
    ignoring(expr);
    return l;
  } catch (conv_error *e) {
    std::cerr << e->to_string() << std::endl;
    ignoring(expr);
    return l;
  }

  formula = Z3_mk_iff(z3_ctx, z3_literal(l), constraint);

  // While we have a literal, don't assert that it's true, only the link
  // between the formula and the literal. Otherwise, we risk asserting that a
  // formula within a assertion-statement is true or false.
  assert_formula(formula);

  return l;
}

void
z3_convt::convert_identifier_pointer(const expr2tc &expr, std::string symbol,
                                     Z3_ast &bv)
{
  Z3_type_ast tuple_type;
  Z3_sort native_int_sort;
  std::string cte, identifier;
  unsigned int obj_num;
  bool got_obj_num = false;

  if (int_encoding)
    native_int_sort = Z3_mk_int_sort(z3_ctx);
  else
    native_int_sort = Z3_mk_bv_sort(z3_ctx, config.ansi_c.int_width);

  create_pointer_type(tuple_type);

  if (is_symbol2t(expr)) {
    const symbol2t &sym = to_symbol2t(expr);
    if (sym.name.as_string() == "NULL" || sym.name.as_string() == "0") {
      obj_num = pointer_logic.get_null_object();
      got_obj_num = true;
    }
  }

  if (!got_obj_num)
    // add object won't duplicate objs for identical exprs (it's a map)
    obj_num = pointer_logic.add_object(expr);

  bv = z3_api.mk_var(symbol.c_str(), tuple_type);

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

    type2tc ptr_loc_type(new unsignedbv_type2t(config.ansi_c.int_width));

    std::string start_name = "__ESBMC_ptr_obj_start_" + itos(obj_num);
    std::string end_name = "__ESBMC_ptr_obj_end_" + itos(obj_num);

    expr2tc start_sym(new symbol2t(ptr_loc_type, start_name));
    expr2tc end_sym(new symbol2t(ptr_loc_type, end_name));

    // Another thing to note is that the end var must be /the size of the obj/
    // from start. Express this in irep.
    expr2tc endisequal;
    try {
      uint64_t type_size = expr->type->get_width() / 8;
      expr2tc const_offs(new constant_int2t(ptr_loc_type, BigInt(type_size)));
      expr2tc start_plus_offs(new add2t(ptr_loc_type, start_sym, const_offs));
      endisequal = expr2tc(new equality2t(start_plus_offs, end_sym));
    } catch (array_type2t::dyn_sized_array_excp *e) {
      // Dynamically (nondet) sized array; take that size and use it for the
      // offset-to-end expression.
      const expr2tc size_expr = e->size;
      expr2tc start_plus_offs(new add2t(ptr_loc_type, start_sym, size_expr));
      endisequal = expr2tc(new equality2t(start_plus_offs, end_sym));
    } catch (type2t::symbolic_type_excp *e) {
      // Type is empty or code -- something that we can never have a real size
      // for. In that case, create an object of size 1: this means we have a
      // valid entry in the address map, but that any modification of the
      // pointer leads to invalidness, because there's no size to think about.
      expr2tc const_offs(new constant_int2t(ptr_loc_type, BigInt(1)));
      expr2tc start_plus_offs(new add2t(ptr_loc_type, start_sym, const_offs));
      endisequal = expr2tc(new equality2t(start_plus_offs, end_sym));
    }

    // Also record the amount of memory space we're working with for later usage
    total_mem_space += pointer_offset_size(*expr->type.get()).to_long() + 1;

    // Assert that start + offs == end
    Z3_ast offs_eq;
    convert_bv(endisequal, offs_eq);
    assert_formula(offs_eq);

    // Even better, if we're operating in bitvector mode, it's possible that
    // Z3 will try to be clever and arrange the pointer range to cross the end
    // of the address space (ie, wrap around). So, also assert that end > start
    expr2tc wraparound(new greaterthan2t(end_sym, start_sym));
    Z3_ast wraparound_eq;
    convert_bv(wraparound, wraparound_eq);
    assert_formula(wraparound_eq);

    // We'll place constraints on those addresses later, in finalize_pointer_chain

    addr_space_data[obj_num] =
          pointer_offset_size(*expr->type.get()).to_long() + 1;

    Z3_ast start_ast, end_ast;
    convert_bv(start_sym, start_ast);
    convert_bv(end_sym, end_ast);

    // Actually store into array
    Z3_ast range_tuple = z3_api.mk_var(
                       ("__ESBMC_ptr_addr_range_" + itos(obj_num)).c_str(),
                       addr_space_tuple_sort);
    Z3_ast init_val = z3_api.mk_tuple(addr_space_tuple_sort, start_ast,
                                      end_ast, NULL);
    Z3_ast eq = Z3_mk_eq(z3_ctx, range_tuple, init_val);
    assert_formula(eq);

    // Update array
    bump_addrspace_array(obj_num, range_tuple);

    // Finally, ensure that the array storing whether this pointer is dynamic,
    // is initialized for this ptr to false. That way, only pointers created
    // through malloc will be marked dynamic.

    type2tc arrtype(new array_type2t(type2tc(new bool_type2t()),
                                     expr2tc((expr2t*)NULL), true));
    expr2tc allocarr(new symbol2t(arrtype, dyn_info_arr_name.as_string()));
    Z3_ast allocarray;
    convert_bv(allocarr, allocarray);

    Z3_ast idxnum = Z3_mk_int(z3_ctx, obj_num, native_int_sort);
    Z3_ast select = Z3_mk_select(z3_ctx, allocarray, idxnum);
    Z3_ast isfalse = Z3_mk_eq(z3_ctx, Z3_mk_false(z3_ctx), select);
    assert_formula(isfalse);
  }
}

void
z3_convt::set_to(const expr2tc &expr, bool value)
{

  l_set_to(convert(expr), value);

  if (is_equality2t(expr) && value) {
    const equality2t eq = to_equality2t(expr);
    if (is_union_type(eq.side_1->type) && is_with2t(eq.side_2)) {
      const symbol2t sym = to_symbol2t(eq.side_1);
      const with2t with = to_with2t(eq.side_2);
      const union_type2t &type = to_union_type(eq.side_1->type);
      const std::string &ref = sym.name.as_string();
      const constant_string2t &str = to_constant_string2t(with.update_field);

      unsigned int idx = 0;
      forall_names(it, type.member_names) {
        if (*it == str.value)
          break;
        idx++;
      }

      assert(idx != type.member_names.size() &&
             "Member name of with expr not found in struct/union type");

      union_vars.insert(std::pair<std::string, unsigned int>(ref, idx));
    }
  }

}

literalt
z3_convt::land(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for (unsigned int i = 0; i < bv.size(); i++)
    args[i] = z3_literal(bv[i]);

  result = Z3_mk_and(z3_ctx, bv.size(), args);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

literalt
z3_convt::lor(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  Z3_ast *args = (Z3_ast*)alloca(size * sizeof(Z3_ast));
  Z3_ast result, formula;

  for (unsigned int i = 0; i < bv.size(); i++)
    args[i] = z3_literal(bv[i]);

  result = Z3_mk_or(z3_ctx, bv.size(), args);

  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;
}

literalt
z3_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  literalt l = new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_and(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

literalt
z3_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);
  if (a == b) return a;

  literalt l = new_variable();
  Z3_ast result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = Z3_mk_or(z3_ctx, 2, operand);
  formula = Z3_mk_iff(z3_ctx, z3_literal(l), result);
  assert_formula(formula);

  return l;

}

literalt
z3_convt::lnot(literalt a)
{
  a.invert();

  return a;
}

literalt
z3_convt::limplies(literalt a, literalt b)
{
  return lor(lnot(a), b);
}

literalt
z3_convt::new_variable()
{
  literalt l;

  l.set(_no_variables, false);

  set_no_variables(_no_variables + 1);

  return l;
}

void
z3_convt::eliminate_duplicates(const bvt &bv, bvt &dest)
{
  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin(); it != bv.end(); it++)
  {
    if (s.insert(*it).second)
      dest.push_back(*it);
  }
}

bool
z3_convt::process_clause(const bvt &bv, bvt &dest)
{

  dest.clear();

  // empty clause! this is UNSAT
  if (bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin();
       it != bv.end();
       it++)
  {
    literalt l = *it;

    // we never use index 0
    assert(l.var_no() != 0);

    if (l.is_true())
      return true;  // clause satisfied

    if (l.is_false())
      continue;

    assert(l.var_no() < _no_variables);

    // prevent duplicate literals
    if (s.insert(l).second)
      dest.push_back(l);

    if (s.find(lnot(l)) != s.end())
      return true;  // clause satisfied
  }

  return false;
}

void
z3_convt::lcnf(const bvt &bv)
{

  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.size() == 0)
    return;

  Z3_ast lor_var, *args = (Z3_ast*)alloca(new_bv.size() * sizeof(Z3_ast));
  unsigned int i = 0;

  for (bvt::const_iterator it = new_bv.begin(); it != new_bv.end(); it++, i++)
    args[i] = z3_literal(*it);

  if (i > 1) {
    lor_var = Z3_mk_or(z3_ctx, i, args);
    assert_formula(lor_var);
  } else   {
    assert_formula(args[0]);
  }
}

Z3_ast
z3_convt::z3_literal(literalt l)
{

  Z3_ast literal_l;
  std::string literal_s;

  if (l == const_literal(false))
    return Z3_mk_false(z3_ctx);
  else if (l == const_literal(true))
    return Z3_mk_true(z3_ctx);

  literal_s = "l" + i2string(l.var_no());
  literal_l = z3_api.mk_bool_var(literal_s.c_str());

  if (l.sign()) {
    return Z3_mk_not(z3_ctx, literal_l);
  }

  return literal_l;
}

tvt
z3_convt::l_get(literalt a) const
{
  tvt result = tvt(tvt::TV_ASSUME);
  std::string literal;

  if (a.is_true()) {
    return tvt(true);
  } else if (a.is_false())    {
    return tvt(false);
  }

  symbol_exprt sym("l" + i2string(a.var_no()), bool_typet());
  exprt res = get(sym);

  if (res.is_true())
    result = tvt(true);
  else if (res.is_false())
    result = tvt(false);
  else
    result = tvt(tvt::TV_UNKNOWN);

  if (a.sign())
    result = !result;

  return result;
}

void
z3_convt::assert_formula(Z3_ast ast, bool needs_literal)
{

  // If we're not going to be using the assumptions (ie, for unwidening and for
  // smtlib) then just assert the fact to be true.
  if (!store_assumptions) {
    Z3_assert_cnstr(z3_ctx, ast);
    return;
  }

  if (!needs_literal) {
    Z3_assert_cnstr(z3_ctx, ast);
    assumpt.push_front(ast);
  } else {
    literalt l = new_variable();
    Z3_ast formula = Z3_mk_iff(z3_ctx, z3_literal(l), ast);
    Z3_assert_cnstr(z3_ctx, formula);

    if (smtlib)
      assumpt.push_front(ast);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}

void
z3_convt::assert_literal(literalt l, Z3_ast formula)
{

  Z3_assert_cnstr(z3_ctx, formula);
  if (store_assumptions) {
    if (smtlib)
      assumpt.push_front(formula);
    else
      assumpt.push_front(z3_literal(l));
  }

  return;
}

Z3_context z3_convt::z3_ctx = NULL;

unsigned int z3_convt::num_ctx_ileaves = 0;
bool z3_convt::s_is_uw = false;
