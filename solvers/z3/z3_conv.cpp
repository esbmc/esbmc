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
#include <base_type.h>

#include "z3_conv.h"
#include "../ansi-c/c_types.h"

#define cast_to_z3(arg) (*(reinterpret_cast<z3::expr *&>((arg))))
#define cast_to_z3_sort(arg) (*(reinterpret_cast<z3::sort *>((arg))))

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                          "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

extern void finalize_symbols(void);

Z3_ast workaround_Z3_mk_bvadd_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvadd_no_underflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_overflow(Z3_context ctx, Z3_ast a1,Z3_ast a2);
Z3_ast workaround_Z3_mk_bvsub_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                          Z3_bool is_signed);
Z3_ast workaround_Z3_mk_bvneg_no_overflow(Z3_context ctx, Z3_ast a);
z3_convt::z3_convt(bool int_encoding, bool is_cpp, const namespacet &_ns)
: smt_convt(), ns(_ns)
{
  this->int_encoding = int_encoding;

  assumpt_mode = false;
  no_variables = 1;
  level_ctx = 0;

  z3::config conf;
  conf.set("MODEL", true);
  conf.set("RELEVANCY", 0);
  conf.set("SOLVER", true);
  // Disabling this option results in the enablement of --symex-thread-guard on
  // 03_exor_01 to not explode solving time. No idea why this is the case,
  // doesn't affect any other solving time.
  conf.set("ARRAY_ALWAYS_PROP_UPWARD", false);

  ctx.init(conf, int_encoding);

  z3_ctx = ctx;
  Z3_set_ast_print_mode(z3_ctx, Z3_PRINT_SMTLIB_COMPLIANT);

  solver = z3::solver(ctx);

  setup_pointer_sort();
  pointer_logic.push_back(pointer_logict());
  addr_space_sym_num.push_back(0);
  addr_space_data.push_back(std::map<unsigned, unsigned>());
  total_mem_space.push_back(0);

  assumpt_ctx_stack.push_back(assumpt.begin());

  init_addr_space_array();

  // Pick a modelling array to shoehorn initialization data into. Because
  // we don't yet have complete data for whether pointers are dynamic or not,
  // this is the one modelling array that absolutely _has_ to be initialized
  // to false for each element, which is going to be shoved into
  // convert_identifier_pointer.
  if (is_cpp) {
    dyn_info_arr_name = "cpp::__ESBMC_is_dynamic&0#1";
  } else {
    dyn_info_arr_name = "c::__ESBMC_is_dynamic&0#1";
  }

  // Pre-seed type cache with a few values that might not go in due to
  // specialised code paths.
  sort_cache.insert(std::pair<const type2tc, z3::sort>(type_pool.get_bool(),
                    ctx.bool_sort()));
}


z3_convt::~z3_convt()
{

  // jmorse - remove when smtlib printer exists and works.
#if 0
  if (smtlib) {
    std::ofstream temp_out;
    Z3_string smt_lib_str, logic;
    Z3_ast assumpt_array_ast[assumpt.size() + 1];
    z3::expr formula;
    formula = ctx.bool_val(true);

    std::list<z3::expr>::const_iterator it;
    unsigned int i;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumpt_array_ast[i] = *it;
    }

    if (int_encoding)
      logic = "QF_AUFLIRA";
    else
      logic = "QF_AUFBV";

    smt_lib_str = Z3_benchmark_to_smtlib_string(z3_ctx, "ESBMC", logic,
                                    "unknown", "", assumpt.size(),
                                    assumpt_array_ast, formula);

    temp_out.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);

    temp_out << smt_lib_str << std::endl;
  }
#endif
}

void
z3_convt::push_ctx(void)
{

  smt_convt::push_ctx();
  intr_push_ctx();
  solver.push();
}

void
z3_convt::pop_ctx(void)
{

  solver.pop();
  intr_pop_ctx();
  smt_convt::pop_ctx();;

  // Clear model if we have one.
  model = z3::model();
}

void
z3_convt::intr_push_ctx(void)
{

  // Also push/duplicate pointer logic state.
  pointer_logic.push_back(pointer_logic.back());
  addr_space_sym_num.push_back(addr_space_sym_num.back());
  addr_space_data.push_back(addr_space_data.back());
  total_mem_space.push_back(total_mem_space.back());

  // Store where we are in the list of assumpts.
  std::list<z3::expr>::iterator it = assumpt.end();
  it--;
  assumpt_ctx_stack.push_back(it);
}

void
z3_convt::intr_pop_ctx(void)
{

  // Erase everything on stack since last push_ctx
  std::list<z3::expr>::iterator it = assumpt_ctx_stack.back();
  ++it;
  assumpt.erase(it, assumpt.end());
  assumpt_ctx_stack.pop_back();

  bv_cachet::nth_index<1>::type &cache_numindex = bv_cache.get<1>();
  cache_numindex.erase(ctx_level);

  pointer_logic.pop_back();
  addr_space_sym_num.pop_back();
  addr_space_data.pop_back();
  total_mem_space.pop_back();
}

void
z3_convt::init_addr_space_array(void)
{
  z3::symbol mk_tuple_name, proj_names[2];
  Z3_symbol proj_names_sym[2];
  Z3_sort proj_types[2];
  Z3_func_decl mk_tuple_decl, proj_decls[2];

  addr_space_sym_num.back() = 1;

  // Place locations of numerical addresses for null and invalid_obj.

  z3::expr tmp =
    ctx.constant("__ESBMC_ptr_obj_start_0", ctx.esbmc_int_sort());
  z3::expr num = ctx.esbmc_int_val(0);
  z3::expr eq = tmp == num;

  assert_formula(eq);

  tmp = ctx.constant("__ESBMC_ptr_obj_end_0", ctx.esbmc_int_sort());
  num = ctx.esbmc_int_val(0);
  eq = tmp == num;

  assert_formula(eq);

  tmp = ctx.constant("__ESBMC_ptr_obj_start_1", ctx.esbmc_int_sort());
  num = ctx.esbmc_int_val(1);
  eq = tmp == num;
  assert_formula(eq);

  tmp = ctx.constant("__ESBMC_ptr_obj_end_1", ctx.esbmc_int_sort());
  num = ctx.esbmc_int_val((uint64_t)0xFFFFFFFFFFFFFFFFULL);
  eq = tmp == num;
  assert_formula(eq);

  z3::sort tmp_proj_type = ctx.esbmc_int_sort();
  proj_types[0] = proj_types[1] = tmp_proj_type;

  mk_tuple_name = z3::symbol(ctx, "struct_type_addr_space_tuple");
  proj_names[0] = z3::symbol(ctx, "start");
  proj_names[1] = z3::symbol(ctx, "end");
  proj_names_sym[0] = proj_names[0];
  proj_names_sym[1] = proj_names[1];

  addr_space_tuple_sort = z3::to_sort(ctx, Z3_mk_tuple_sort(
                                      ctx, mk_tuple_name, 2,
                                      proj_names_sym, proj_types,
                                      &mk_tuple_decl, proj_decls));
  Z3_func_decl tmp_addr_space_decl =
    Z3_get_tuple_sort_mk_decl(ctx, addr_space_tuple_sort);
  addr_space_tuple_decl = z3::func_decl(ctx, tmp_addr_space_decl);

  // Generate initial array with all zeros for all fields.
  addr_space_arr_sort = 
                  ctx.array_sort(ctx.esbmc_int_sort(), addr_space_tuple_sort);

  num = ctx.esbmc_int_val(0);

  z3::expr initial_val =
    addr_space_tuple_decl.make_tuple("", &num, &num, NULL);

  z3::expr initial_const = z3::const_array(ctx.esbmc_int_sort(), initial_val);
  z3::expr first_name =
    ctx.constant("__ESBMC_addrspace_arr_0", addr_space_arr_sort);

  eq = first_name == initial_const;
  assert_formula(eq);

  z3::expr range_tuple =
    ctx.constant("__ESBMC_ptr_addr_range_0", addr_space_tuple_sort);
  initial_val = addr_space_tuple_decl.make_tuple("", &num, &num, NULL);

  eq = initial_val == range_tuple;
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.back().get_null_object(), range_tuple);

  // We also have to initialize the invalid object... however, I've no idea
  // what it /means/ yet, so go for some arbitary value.
  num = ctx.esbmc_int_val(1);
  range_tuple = ctx.constant("__ESBMC_ptr_addr_range_1",
                              addr_space_tuple_sort);
  initial_val = addr_space_tuple_decl.make_tuple("", &num, &num, NULL);
  eq = initial_val == range_tuple;
  assert_formula(eq);

  bump_addrspace_array(pointer_logic.back().get_invalid_object(), range_tuple);

  // Associate the symbol "0" with the null object; this is necessary because
  // of the situation where 0 is valid as a representation of null, but the
  // frontend (for whatever reasons) converts it to a symbol rather than the
  // way it handles NULL (constant with val "NULL")
  z3::expr zero_sym = ctx.constant("0", pointer_sort);

  z3::expr zero_int= ctx.esbmc_int_val(0);
  z3::expr ptr_val = pointer_decl(zero_int, zero_int);
  z3::expr constraint = zero_sym == ptr_val;
  assert_formula(constraint);

  // Do the same thing, for the name "NULL".
  z3::expr null_sym = ctx.constant("NULL", pointer_sort);
  constraint = null_sym == ptr_val;
  assert_formula(constraint);

  // And for the "INVALID" object (which we're issuing with a name now), have
  // a pointer object num of 1, and a free pointer offset. Anything of worth
  // using this should extract only the object number.

  z3::expr args[2];
  args[0] = ctx.esbmc_int_val(1);
  args[1] = ctx.fresh_const(NULL, pointer_sort);
  z3::expr invalid = mk_tuple_update(args[1], 0, args[0]);
  z3::expr invalid_name = ctx.constant("INVALID", pointer_sort);
  constraint = invalid == invalid_name;
  assert_formula(constraint);

  // Record the fact that we've registered these objects
  addr_space_data.back()[0] = 0;
  addr_space_data.back()[1] = 0;

  return;
}

void
z3_convt::bump_addrspace_array(unsigned int idx, const z3::expr &val)
{
  std::string str, new_str;

  str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num.back()++);
  z3::expr addr_sym = ctx.constant(str.c_str(), addr_space_arr_sort);
  z3::expr obj_idx = ctx.esbmc_int_val(idx);

  z3::expr store = z3::store(addr_sym, obj_idx, val);

  new_str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num.back());
  z3::expr new_addr_sym = ctx.constant(new_str.c_str(), addr_space_arr_sort);

  z3::expr eq = new_addr_sym == store;
  assert_formula(eq);

  return;
}

std::string
z3_convt::get_cur_addrspace_ident(void)
{

  std::string str = "__ESBMC_addrspace_arr_" + itos(addr_space_sym_num.back());
  return str;
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
z3_convt::finalize_pointer_chain(unsigned int objnum)
{
  unsigned int num_ptrs = addr_space_data.back().size();
  if (num_ptrs == 0)
    return;

  // Floating model - we assert that all objects don't overlap each other,
  // but otherwise their locations are entirely defined by Z3. Inefficient,
  // but necessary for accuracy. Unfortunately, has high complexity (O(n^2))

  // Implementation: iterate through all objects; assert that those with lower
  // object nums don't overlap the current one. So for every particular pair
  // of object numbers in the set there'll be a doesn't-overlap clause.

  z3::expr i_start = ctx.constant(
                       ("__ESBMC_ptr_obj_start_" + itos(objnum)).c_str(),
                       ctx.esbmc_int_sort());
  z3::expr i_end = ctx.constant(
                       ("__ESBMC_ptr_obj_end_" + itos(objnum)).c_str(),
                       ctx.esbmc_int_sort());

  for (unsigned j = 0; j < objnum; j++) {
    // Obj 1 is designed to overlap
    if (j == 1)
      continue;

    z3::expr j_start = ctx.constant(
                       ("__ESBMC_ptr_obj_start_" + itos(j)).c_str(),
                       ctx.esbmc_int_sort());
    z3::expr j_end = ctx.constant(
                       ("__ESBMC_ptr_obj_end_" + itos(j)).c_str(),
                       ctx.esbmc_int_sort());

    // Formula: (i_end < j_start) || (i_start > j_end)
    // Previous assertions ensure start < end for all objs.
    // Hey hey, I can just write that with the C++y api!
    z3::expr formula;
    formula = (mk_lt(i_end, j_start, true)) || (mk_gt(i_start, j_end, true));
    assert_formula(formula);
  }

  return;
}

prop_convt::resultt
z3_convt::dec_solve(void)
{
  unsigned major, minor, build, revision;
  z3::check_result result;
  Z3_get_version(&major, &minor, &build, &revision);

  std::cout << "Solving with SMT Solver Z3 v" << major << "." << minor << "\n";

  result = check2_z3_properties();

  if (result == z3::unsat)
    return prop_convt::P_UNSATISFIABLE;
  else if (result == z3::unknown)
    return prop_convt::P_ERROR;
  else
    return prop_convt::P_SATISFIABLE;
}

z3::check_result
z3_convt::check2_z3_properties(void)
{
  z3::check_result result;
  unsigned i;
  std::string literal;
  z3::expr_vector assumptions(ctx);

  if (assumpt_mode) {
    std::list<z3::expr>::const_iterator it;
    for (it = assumpt.begin(), i = 0; it != assumpt.end(); it++, i++) {
      assumptions.push_back(*it);
    }
  }

  // XXX XXX XXX jmorse: as of 5dd8a432 running with --smt-during-symex on tests
  // like 03_exor_01 caused a significant performance hit for no known reason.
  // Solving got progressively slower as more interleavings were checked.
  // Profiling said a lot of time was spent in Z3's
  // bv_simplifier_plugin::bit2bool_simplify method. This doesn't happen if you
  // run with no additional options. No idea why, but the belief is that the
  // solver is caching something, bloats, and leads to a performance hit.
  //
  // So during debugging I added the following line to see whether some asserts
  // were being left in the solver accidentally leading to the bloat and... it
  // just stopped. Presumably this accidentally flushes some kind of internal
  // cache and kills bloatage; I've no idea why; but if you remove it there's
  // a significant performance hit.
  z3::expr_vector vec = solver.assertions();

  if (assumpt_mode) {
    result = solver.check(assumptions);
  } else {
    result = solver.check();
  }

  if (result == z3::sat)
    model = solver.get_model();

  if (config.options.get_bool_option("dump-z3-assigns") && result == z3::sat)
    std::cout << Z3_model_to_string(z3_ctx, model);

  return result;
}

void
z3_convt::convert_smt_type(const bool_type2t &type __attribute__((unused)),
                           void *_bv)
{
  z3::sort &sort = cast_to_z3_sort(_bv);

  sort = ctx.bool_sort();
  return;
}

void
z3_convt::convert_smt_type(const unsignedbv_type2t &type, void *_bv)
{
  z3::sort &sort = cast_to_z3_sort(_bv);

  if (int_encoding) {
    sort = ctx.esbmc_int_sort();
  } else {
    unsigned int width = type.get_width();
    sort = ctx.bv_sort(width);
  }

  return;
}

void
z3_convt::convert_smt_type(const signedbv_type2t &type, void *_bv)
{
  z3::sort &sort = cast_to_z3_sort(_bv);

  if (int_encoding) {
    sort = ctx.esbmc_int_sort();
  } else {
    unsigned int width = type.get_width();
    sort = ctx.bv_sort(width);
  }

  return;
}

void
z3_convt::convert_smt_type(const array_type2t &type, void *_bv)
{
  z3::sort &sort = cast_to_z3_sort(_bv), elem_sort;

  convert_type(type.subtype, elem_sort);
  sort = ctx.array_sort(ctx.esbmc_int_sort(), elem_sort);

  return;
}

void
z3_convt::convert_smt_type(const pointer_type2t &type __attribute__((unused)),
                           void *_bv)
{
  // Storage for Z3 objects that keep a reference,
  z3::sort int_sort;
  z3::sort &sort = cast_to_z3_sort(_bv);
  z3::symbol tuple_name;
  z3::symbol proj_name_refs[2];
  // Copies of the above, in a form that can be passed directly the the C api.
  Z3_func_decl mk_tuple_decl, proj_decls[2];
  Z3_symbol proj_names[2];
  Z3_sort proj_types[2];

  tuple_name = z3::symbol(ctx, "pointer_tuple");
  int_sort = ctx.esbmc_int_sort();
  proj_types[0] = proj_types[1] = int_sort;

  proj_name_refs[0] = z3::symbol(ctx, "object");
  proj_name_refs[1] = z3::symbol(ctx, "index");
  proj_names[0] = proj_name_refs[0];
  proj_names[1] = proj_name_refs[1];

  sort = z3::to_sort(ctx, Z3_mk_tuple_sort(ctx, tuple_name, 2, proj_names,
                                       proj_types, &mk_tuple_decl, proj_decls));
  return;
}

void
z3_convt::convert_struct_union_type(const std::vector<type2tc> &members,
                                    const std::vector<irep_idt> &member_names,
                                    const irep_idt &struct_name, bool uni,
                                    void *_bv)
{
  z3::symbol mk_tuple_name, *proj_names;
  z3::sort *proj_types;
  z3::sort &sort = cast_to_z3_sort(_bv);
  Z3_func_decl mk_tuple_decl, *proj_decls;
  std::string name;
  u_int num_elems;

  num_elems = members.size();
  if (uni)
    num_elems++;

  proj_names = new z3::symbol[num_elems];
  proj_types = new z3::sort[num_elems];
  proj_decls = new Z3_func_decl[num_elems];

  name = ((uni) ? "union" : "struct" );
  name += "_type_" + struct_name.as_string();
  mk_tuple_name = z3::symbol(ctx, name.c_str());

  if (!members.size()) {
    sort = z3::to_sort(ctx, Z3_mk_tuple_sort(ctx, mk_tuple_name, 0, NULL, NULL, &mk_tuple_decl, NULL));
    return;
  }

  u_int i = 0;
  std::vector<irep_idt>::const_iterator mname = member_names.begin();
  for (std::vector<type2tc>::const_iterator it = members.begin();
       it != members.end(); it++, mname++, i++)
  {
    proj_names[i] = z3::symbol(ctx, mname->as_string().c_str());
    convert_type(*it, proj_types[i]);
  }

  if (uni) {
    // ID field records last value written to union
    proj_names[num_elems - 1] = z3::symbol(ctx, "id");
    // XXXjmorse - must this field really become a bitfield, ever? It's internal
    // tracking data, not program data.
    proj_types[num_elems - 1] = ctx.esbmc_int_sort();
  }

  // Unpack pointers from Z3++ objects.
  Z3_symbol *unpacked_symbols = new Z3_symbol[num_elems];
  Z3_sort *unpacked_sorts = new Z3_sort[num_elems];
  for (i = 0; i < num_elems; i++) {
    unpacked_symbols[i] = proj_names[i];
    unpacked_sorts[i] = proj_types[i];
  }

  sort = z3::to_sort(ctx, Z3_mk_tuple_sort(ctx, mk_tuple_name, num_elems,
                           unpacked_symbols, unpacked_sorts, &mk_tuple_decl,
                           proj_decls));

  delete[] unpacked_symbols;
  delete[] unpacked_sorts;
  delete[] proj_names;
  delete[] proj_types;
  delete[] proj_decls;

  return;
}

void
z3_convt::convert_smt_type(const struct_type2t &type, void *_bv)
{

  convert_struct_union_type(type.members, type.member_names, type.name,
                            false, _bv);
  return;
}

void
z3_convt::convert_smt_type(const union_type2t &type, void *_bv)
{

  convert_struct_union_type(type.members, type.member_names, type.name,
                            true, _bv);
  return;
}

void
z3_convt::convert_smt_type(const fixedbv_type2t &type, void *_bv)
{
  z3::sort &sort = cast_to_z3_sort(_bv);

  unsigned int width = type.get_width();

  if (int_encoding)
    sort = ctx.real_sort();
  else
    sort = ctx.bv_sort(width);

  return;
}

void
z3_convt::setup_pointer_sort(void)
{
  z3::symbol proj_names[2], mk_tuple_name;
  z3::sort proj_type;
  Z3_symbol proj_names_ref[2];
  Z3_sort sort_arr[2];
  Z3_func_decl mk_tuple_decl, proj_decls[2];

  proj_type = ctx.esbmc_int_sort();
  sort_arr[0] = sort_arr[1] = proj_type;

  mk_tuple_name = z3::symbol(ctx, "pointer_tuple");
  proj_names[0] = z3::symbol(ctx, "object");
  proj_names[1] = z3::symbol(ctx, "index");
  proj_names_ref[0] = proj_names[0];
  proj_names_ref[1] = proj_names[1];

  z3::sort s = z3::to_sort(ctx,
              Z3_mk_tuple_sort(ctx, mk_tuple_name, 2, proj_names_ref, sort_arr,
                               &mk_tuple_decl, proj_decls));

  pointer_sort = z3::to_sort(ctx, s);
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(ctx, s);
  pointer_decl = z3::func_decl(ctx, decl);
  return;
}

void
z3_convt::convert_smt_expr(const symbol2t &sym, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  // References to unsigned int identifiers need to be assumed to be > 0,
  // otherwise the solver is free to assign negative nums to it.
  if (is_unsignedbv_type(sym.type) && int_encoding) {
    output = ctx.constant((sym.get_symbol_name().c_str()), ctx.int_sort());
    z3::expr formula = mk_ge(output, ctx.int_val(0), true);
    assert_formula(formula);
    return;
  }

  z3::sort sort;
  convert_type(sym.type, sort);
  output = ctx.constant(sym.get_symbol_name().c_str(), sort);
}

void
z3_convt::convert_smt_expr(const constant_int2t &sym, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  unsigned int bitwidth = sym.type->get_width();

  if (is_unsignedbv_type(sym.type)) {
    output = ctx.esbmc_int_val(sym.as_ulong(), bitwidth);
  } else {
    assert(is_signedbv_type(sym.type));
    output = ctx.esbmc_int_val(sym.as_long(), bitwidth);
  }

  return;
}

void
z3_convt::convert_smt_expr(const constant_fixedbv2t &sym, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  unsigned int bitwidth = sym.type->get_width();

  assert(is_fixedbv_type(sym.type));

  std::string theval = sym.value.to_expr().value().as_string();

  if (int_encoding) {
    std::string result = fixed_point(theval, bitwidth);
    output = ctx.real_val(result.c_str());
  } else {
    z3::expr magnitude, fraction;
    std::string m, f, c;
    m = extract_magnitude(theval, bitwidth);
    f = extract_fraction(theval, bitwidth);
    magnitude = ctx.esbmc_int_val(m.c_str(), bitwidth / 2);
    fraction = ctx.esbmc_int_val(f.c_str(), bitwidth / 2);
    output = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, magnitude, fraction));
  }

  return;
}

void
z3_convt::convert_smt_expr(const constant_bool2t &b, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  output = ctx.bool_val(b.constant_value);
}

void
z3_convt::convert_struct_union(const std::vector<expr2tc> &members,
                               const std::vector<type2tc> &member_types,
                               const type2tc &type, bool is_union, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  // Converts a static struct/union - IE, one that hasn't had any "with"
  // operations applied to it, perhaps due to initialization or constant
  // propagation.
  u_int i = 0;

  z3::sort sort;
  convert_type(type, sort);

  unsigned size = member_types.size();
  if (is_union)
    size++;

  z3::expr *args = new z3::expr[size];

  unsigned int numoperands = members.size();
  // Populate tuple with members of that struct/union
  forall_types(it, member_types) {
    if (i < numoperands) {
      convert_bv(members[i], args[i]);
    } else {
      // Turns out that unions don't necessarily initialize all members.
      // If no initialization give, use free (fresh) variable.
      z3::sort s;
      convert_type(*it, s);
      args[i] = ctx.fresh_const(NULL, s);
    }

    i++;
  }

  // Update unions "last-set" member to be the last field
  if (is_union)
    args[size-1] = ctx.esbmc_int_val(i);

  // Create tuple itself, return to caller. This is a lump of data, we don't
  // need to bind it to a name or symbol.
  Z3_func_decl decl = Z3_get_tuple_sort_mk_decl(ctx, sort);
  z3::func_decl d(ctx, decl);
  output = d.make_tuple_from_array(size, args);
  delete[] args;
}

void
z3_convt::convert_smt_expr(const constant_struct2t &data, void *_bv)
{
  const struct_type2t &ref =
    dynamic_cast<const struct_type2t&>(*data.type.get());
  convert_struct_union(data.datatype_members, ref.members, data.type,
                       false, _bv);
}

void
z3_convt::convert_smt_expr(const constant_union2t &data, void *_bv)
{
  const union_type2t &ref = dynamic_cast<const union_type2t&>(*data.type.get());
  convert_struct_union(data.datatype_members, ref.members, data.type,
                       true, _bv);
}

void
z3_convt::convert_smt_expr(const constant_array2t &array, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  u_int i = 0;
  z3::sort z3_array_type;
  z3::expr int_cte, val_cte;
  z3::sort elem_type;

  const array_type2t &arr_type = to_array_type(array.type);
  convert_type(arr_type.subtype, elem_type);
  z3_array_type = ctx.array_sort(ctx.esbmc_int_sort(), elem_type);

  output = ctx.fresh_const(NULL, z3_array_type);

  i = 0;
  forall_exprs(it, array.datatype_members) {
    int_cte = ctx.esbmc_int_val(i);

    convert_bv(*it, val_cte);

    output = z3::store(output, int_cte, val_cte);
    ++i;
  }
}

void
z3_convt::convert_smt_expr(const constant_array_of2t &array, void *_bv)
{
  z3::expr value, index;
  z3::sort array_type;
  std::string tmp, identifier;
  int64_t size;
  u_int j;
  z3::expr &output = cast_to_z3(_bv);

  const array_type2t &arr = to_array_type(array.type);

  convert_type(array.type, array_type);

  if (arr.size_is_infinite) {
    // Don't attempt to do anything with this. The user is on their own.
    output = ctx.fresh_const(NULL, array_type);
    return;
  }

  assert(is_constant_int2t(arr.array_size) &&
         "array_of sizes should be constant");

  const constant_int2t &sz = to_constant_int2t(arr.array_size);
  size = sz.as_long();

  convert_bv(array.initializer, value);

  if (is_bool_type(arr.subtype)) {
    value = ctx.bool_val(false);
  }

  output = ctx.fresh_const(NULL, array_type);

  //update array
  for (j = 0; j < size; j++)
  {
    index = ctx.esbmc_int_val(j);
    output = z3::store(output, index, value);
  }
}

void
z3_convt::convert_smt_expr(const constant_string2t &str, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  // Convert to array; convert array.
  expr2tc newarray = str.to_array();
  convert_bv(newarray, output);
  return;
}

void
z3_convt::convert_smt_expr(const if2t &ifirep, void *_bv)
{
  z3::expr operand0, operand1, operand2;
  z3::expr &output = cast_to_z3(_bv);

  convert_bv(ifirep.cond, operand0);
  convert_bv(ifirep.true_value, operand1);
  convert_bv(ifirep.false_value, operand2);

  output = z3::ite(operand0, operand1, operand2);
  return;
}

void
z3_convt::convert_smt_expr(const equality2t &equality, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr side1, side2;

  convert_bv(equality.side_1, side1);
  convert_bv(equality.side_2, side2);

  output = side1 == side2;
}

void
z3_convt::convert_smt_expr(const notequal2t &notequal, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr side1, side2;

  convert_bv(notequal.side_1, side1);
  convert_bv(notequal.side_2, side2);

  output = side1 != side2;
}

void
z3_convt::convert_rel(const expr2tc &side1, const expr2tc &side2,
                      ast_convert_calltype_new convert, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  // 6.3.8 defines relation operators on pointers to be comparisons on their
  // bit representation, with pointers to array/struct/union fields comparing
  // as you might expect.
  if (is_pointer_type(side1->type)) {
    expr2tc cast(new typecast2t(uint_type2(), side1));
    convert_bv(cast, args[0]);
  }

  if (is_pointer_type(side2->type)) {
    expr2tc cast(new typecast2t(uint_type2(), side2));
    convert_bv(cast, args[1]);
  }

  output = convert(args[0], args[1], !is_signedbv_type(side1->type));
}

void
z3_convt::convert_smt_expr(const lessthan2t &lessthan, void *_bv)
{
  convert_rel(lessthan.side_1, lessthan.side_2, &z3::mk_lt, _bv);
}

void
z3_convt::convert_smt_expr(const greaterthan2t &greaterthan, void *_bv)
{
  convert_rel(greaterthan.side_1, greaterthan.side_2, &z3::mk_gt, _bv);
}

void
z3_convt::convert_smt_expr(const lessthanequal2t &le, void *_bv)
{
  convert_rel(le.side_1, le.side_2, &z3::mk_le, _bv);
}

void
z3_convt::convert_smt_expr(const greaterthanequal2t &ge, void *_bv)
{
  convert_rel(ge.side_1, ge.side_2, &z3::mk_ge, _bv);
}

void
z3_convt::convert_smt_expr(const not2t &notval, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr z3val;

  convert_bv(notval.value, z3val);
  output = !z3val;
}

void
z3_convt::convert_logic_2ops(const expr2tc &side1, const expr2tc &side2,
                      ast_logic_convert convert, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  output = convert(args[0], args[1]);
}

void
z3_convt::convert_smt_expr(const and2t &andval, void *_bv)
{
  convert_logic_2ops(andval.side_1, andval.side_2, &z3::mk_and, _bv);
}

void
z3_convt::convert_smt_expr(const or2t &orval, void *_bv)
{
  convert_logic_2ops(orval.side_1, orval.side_2, &z3::mk_or, _bv);
}

void
z3_convt::convert_smt_expr(const xor2t &xorval, void *_bv)
{
  convert_logic_2ops(xorval.side_1, xorval.side_2, &z3::mk_xor, _bv);
}

void
z3_convt::convert_smt_expr(const implies2t &implies, void *_bv)
{
  convert_logic_2ops(implies.side_1, implies.side_2, &z3::mk_implies, _bv);
}

void
z3_convt::convert_binop(const expr2tc &side1, const expr2tc &side2,
                        const type2tc &type, ast_logic_convert convert,
                        void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr args[2];

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  // XXXjmorse - int2bv trainwreck.
  if (int_encoding) {
    unsigned int width = side1->type->get_width();
    args[0] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, args[0]));
    width = side1->type->get_width();
    args[1] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, args[1]));
  }

  output = convert(args[0], args[1]);

  if (int_encoding) {
    if (is_signedbv_type(type)) {
      output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, true));
    } else {
      assert(is_unsignedbv_type(type));
      output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, false));
    }
  }
}

void
z3_convt::convert_smt_expr(const bitand2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvand, _bv);
}

void
z3_convt::convert_smt_expr(const bitor2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvor, _bv);
}

void
z3_convt::convert_smt_expr(const bitxor2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvxor, _bv);
}

void
z3_convt::convert_smt_expr(const bitnand2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvnand, _bv);
}

void
z3_convt::convert_smt_expr(const bitnor2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvnor, _bv);
}

void
z3_convt::convert_smt_expr(const bitnxor2t &bitval, void *_bv)
{
  convert_binop(bitval.side_1, bitval.side_2, bitval.type, &z3::mk_bvxnor, _bv);
}

void
z3_convt::convert_smt_expr(const bitnot2t &bitval, void *_bv)
{
  z3::expr arg;
  z3::expr &output = cast_to_z3(_bv);
  convert_bv(bitval.value, arg);
  output = ~arg;
}

void
z3_convt::convert_smt_expr(const lshr2t &bitval, void *_bv)
{
  convert_shift(bitval, bitval.side_1, bitval.side_2, Z3_mk_bvlshr, _bv);
}

void
z3_convt::convert_smt_expr(const neg2t &neg, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr arg;

  convert_bv(neg.value, arg);

  output = -arg;
}

void
z3_convt::convert_smt_expr(const abs2t &abs, void *_bv)
{
  type2tc sign;
  expr2tc zero;
  z3::expr &output = cast_to_z3(_bv);

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
  convert_bv(result, output);
}

void
z3_convt::convert_arith2ops(const expr2tc &side1, const expr2tc &side2,
                            ast_logic_convert convert,
                            void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr args[2];

  if (is_pointer_type(side1->type) ||
      is_pointer_type(side2->type)) {
    std::cerr << "Pointer arithmetic reached convert_arith2ops" << std::endl;
    abort();
  }

  convert_bv(side1, args[0]);
  convert_bv(side2, args[1]);

  output = convert(args[0], args[1]);
}

void
z3_convt::convert_smt_expr(const add2t &add, void *_bv)
{
  if (is_pointer_type(add.type) ||
      is_pointer_type(add.side_1->type) ||
      is_pointer_type(add.side_2->type))
    return convert_pointer_arith(add.expr_id, add.side_1, add.side_2,
                                 add.type, cast_to_z3(_bv));

  convert_arith2ops(add.side_1, add.side_2, &z3::mk_add, _bv);
}

void
z3_convt::convert_smt_expr(const sub2t &sub, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  if (is_pointer_type(sub.type) ||
      is_pointer_type(sub.side_1->type) ||
      is_pointer_type(sub.side_2->type))
    return convert_pointer_arith(sub.expr_id, sub.side_1, sub.side_2,
                                 sub.type, output);

  convert_arith2ops(sub.side_1, sub.side_2, &z3::mk_sub, _bv);
}

void
z3_convt::convert_smt_expr(const mul2t &mul, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  if (is_pointer_type(mul.side_1->type) ||
      is_pointer_type(mul.side_2->type)) {
    std::cerr << "Pointer arithmetic not valid in a multiply" << std::endl;
    abort();
  }

  z3::expr args[2];
  unsigned fraction_bits = 0;

  convert_bv(mul.side_1, args[0]);
  convert_bv(mul.side_2, args[1]);

  if (!is_fixedbv_type(mul.type) || int_encoding) {
    output = args[0] * args[1];
  } else {
    // fixedbv in bv mode. I've no idea if this actually works.
    const fixedbv_type2t &fbvt = to_fixedbv_type(mul.type);
    fraction_bits = fbvt.width - fbvt.integer_bits;
    args[0] = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, fraction_bits, args[0]));
    args[1] = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, fraction_bits, args[1]));
    output = args[0] * args[1];
    output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx,
                                             fbvt.width + fraction_bits - 1,
                                             fraction_bits, output));
  }
}

void
z3_convt::convert_smt_expr(const div2t &div, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  assert(!is_pointer_type(div.type) &&
         !is_pointer_type(div.side_1->type) &&
         !is_pointer_type(div.side_2->type) &&
         "Can't divide pointers");

  z3::expr op0, op1;

  convert_bv(div.side_1, op0);
  convert_bv(div.side_2, op1);

  if (!is_fixedbv_type(div.type) || int_encoding) {
    bool is_unsigned = is_unsignedbv_type(div.side_1->type) ||
                       is_unsignedbv_type(div.side_2->type);
    output = mk_div(op0, op1, is_unsigned);
  } else {
    // Not the foggiest. Copied from convert_div
    assert(is_fixedbv_type(div.type));
    const fixedbv_type2t &fbvt = to_fixedbv_type(div.type);

    unsigned fraction_bits = fbvt.width - fbvt.integer_bits;

    z3::expr zero = ctx.esbmc_int_val(0, fraction_bits);
    z3::expr cat = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, op0, zero));
    z3::expr sext = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, fraction_bits,
                                op1));
    z3::expr div = mk_div(cat, sext, false);
    output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, fbvt.width - 1, 0, div));
  }
}

void
z3_convt::convert_smt_expr(const modulus2t &mod, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  assert(!is_pointer_type(mod.type) &&
         !is_pointer_type(mod.side_1->type) &&
         !is_pointer_type(mod.side_2->type) &&
         "Can't modulus pointers");

  z3::expr op0, op1;

  convert_bv(mod.side_1, op0);
  convert_bv(mod.side_2, op1);

  assert(is_bv_type(mod.type) && "Can only modulus integers");

  if (int_encoding) {
    output = z3::to_expr(ctx, Z3_mk_mod(z3_ctx, op0, op0));
  } else   {
    if (is_signedbv_type(mod.type)) {
      output = z3::to_expr(ctx, Z3_mk_bvsrem(z3_ctx, op0, op1));
    } else if (is_unsignedbv_type(mod.type)) {
      output = z3::to_expr(ctx, Z3_mk_bvurem(z3_ctx, op0, op1));
    }
  }
}

void
z3_convt::convert_shift(const expr2t &shift, const expr2tc &part1,
                        const expr2tc &part2, ast_convert_calltype convert,
                        void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr op0, op1;
  unsigned width_expr, width_op0, width_op1;

  // XXX jmorse - this should feature real integer promotion, and spit out
  // lshr if not in BV mode.

  convert_bv(part1, op0);
  convert_bv(part2, op1);

  width_expr = shift.type->get_width();
  width_op0 = part1->type->get_width();
  width_op1 = part2->type->get_width();

  if (int_encoding) {
    op0 = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op0, op0));
    op1 = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op1, op1));
  }

  if (width_op0 > width_expr)
    op0 = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, (width_expr - 1), 0, op0));
  if (width_op1 > width_expr)
    op1 = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, (width_expr - 1), 0, op1));

  if (width_op0 > width_op1) {
    if (is_unsignedbv_type(part1->type))
      op1 = z3::to_expr(ctx, Z3_mk_zero_ext(z3_ctx, (width_op0 - width_op1), op1));
    else
      op1 = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op1), op1));
  }

  output = z3::to_expr(ctx, convert(z3_ctx, op0, op1));

  if (int_encoding) {
    if (is_signedbv_type(shift.type)) {
      output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, true));
    } else {
      assert(is_unsignedbv_type(shift.type));
      output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, false));
    }
  }
}

void
z3_convt::convert_smt_expr(const shl2t &shl, void *_bv)
{
  convert_shift(shl, shl.side_1, shl.side_2, Z3_mk_bvshl, _bv);
}

void
z3_convt::convert_smt_expr(const ashr2t &ashr, void *_bv)
{
  convert_shift(ashr, ashr.side_1, ashr.side_2, Z3_mk_bvashr, _bv);
}

void
z3_convt::convert_smt_expr(const same_object2t &same, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr pointer[2], objs[2];

  assert(is_pointer_type(same.side_1->type));
  assert(is_pointer_type(same.side_2->type));

  convert_bv(same.side_1, pointer[0]);
  convert_bv(same.side_2, pointer[1]);

  objs[0] = mk_tuple_select(pointer[0], 0);
  objs[1] = mk_tuple_select(pointer[1], 0);
  output = objs[0] == objs[1];
}

void
z3_convt::convert_smt_expr(const pointer_offset2t &offs, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr pointer;

  // See pointer_object2t conversion:
  const expr2tc *ptr = &offs.ptr_obj;
  while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)->type))
    ptr = &to_typecast2t(*ptr).from;

  convert_bv(*ptr, pointer);

  output = mk_tuple_select(pointer, 1); //select pointer offset
}

void
z3_convt::convert_smt_expr(const pointer_object2t &obj, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr pointer;

  // Nix any typecasts; because some operations are generated by malloc
  // assignments, they're given the type of whatever the pointer return type
  // is supposed to be. Which may very well be casted to an integer, which
  // would make the tuple select we're about to make explode.

  const expr2tc *ptr = &obj.ptr_obj;
  while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)->type))
    ptr = &to_typecast2t(*ptr).from;

  convert_bv(*ptr, pointer);

  output = mk_tuple_select(pointer, 0); //select pointer offset
}

void
z3_convt::convert_smt_expr(const address_of2t &obj, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  std::string symbol_name, out;

  if (is_index2t(obj.ptr_obj)) {
    const index2t &idx = to_index2t(obj.ptr_obj);

    if (!is_string_type(idx.source_value->type)) {
      const array_type2t &arr = to_array_type(idx.source_value->type);

      // Pick pointer-to array subtype; need to make pointer arith work.
      expr2tc addrof(new address_of2t(arr.subtype, idx.source_value));
      expr2tc plus(new add2t(addrof->type, addrof, idx.index));
      convert_bv(plus, output);
    } else {
      // Strings; convert with slightly different types.
      type2tc stringtype(new unsignedbv_type2t(8));
      expr2tc addrof(new address_of2t(stringtype, idx.source_value));
      expr2tc plus(new add2t(addrof->type, addrof, idx.index));
      convert_bv(plus, output);
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

    convert_bv(addr, output);

    // Update pointer offset to offset to that field.
    z3::expr num = ctx.esbmc_int_val(offs);
    output = mk_tuple_update(output, 1, num);
  } else if (is_symbol2t(obj.ptr_obj)) {
// XXXjmorse             obj.ptr_obj->expr_id == expr2t::code_id) {

    const symbol2t &symbol = to_symbol2t(obj.ptr_obj);
    convert_identifier_pointer(obj.ptr_obj, symbol.get_symbol_name(), output);
  } else if (is_constant_string2t(obj.ptr_obj)) {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    const constant_string2t &str = to_constant_string2t(obj.ptr_obj);
    std::string identifier =
      "address_of_str_const(" + str.value.as_string() + ")";
    convert_identifier_pointer(obj.ptr_obj, identifier, output);
  } else if (is_if2t(obj.ptr_obj)) {
    // We can't nondeterministically take the address of something; So instead
    // rewrite this to be if (cond) ? &a : &b;.

    const if2t &ifval = to_if2t(obj.ptr_obj);

    expr2tc addrof1(new address_of2t(obj.type, ifval.true_value));
    expr2tc addrof2(new address_of2t(obj.type, ifval.false_value));
    expr2tc newif(new if2t (obj.type, ifval.cond, addrof1, addrof2));
    convert_bv(newif, output);
  } else if (is_typecast2t(obj.ptr_obj)) {
    // Take the address of whatevers being casted. Either way, they all end up
    // being of a pointer_tuple type, so this should be fine.
    address_of2tc tmp(new address_of2t(type2tc(),
                                       to_typecast2t(obj.ptr_obj).from));
    tmp.get()->type = obj.type;
    convert_bv(tmp, output);
  } else {
    throw new conv_error("Unrecognized address_of operand");
  }
}

void
z3_convt::convert_smt_expr(const byte_extract2t &data, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

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

  z3::expr source;

  convert_bv(data.source_value, source);

  if (int_encoding) {
    if (is_fixedbv_type(data.source_value->type)) {
      if (is_bv_type(data.type)) {
        z3::expr tmp;
	source = z3::to_expr(ctx, Z3_mk_real2int(z3_ctx, source));
	tmp = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, source));
	output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, upper, lower, tmp));
	if (is_signedbv_type(data.type))
	  output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, 1));
	else
	  output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, 0));
      } else {
	throw new conv_error("unsupported type for byte_extract");
      }
    } else if (is_bv_type(data.source_value->type)) {
      z3::expr tmp;
      tmp = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, source));

      if (width >= upper)
	output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, upper, lower, tmp));
      else
	output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, upper - lower, 0, tmp));

      if (is_signedbv_type(data.source_value->type))
	output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, 1));
      else
	output = z3::to_expr(ctx, Z3_mk_bv2int(z3_ctx, output, 0));
    } else {
      throw new conv_error("unsupported type for byte_extract");
    }
  } else {
    if (is_struct_type(data.source_value->type)) {
      const struct_type2t &struct_type =to_struct_type(data.source_value->type);
      unsigned i = 0, num_elems = struct_type.members.size();
      z3::expr struct_elem[num_elems + 1], struct_elem_inv[num_elems + 1];

      forall_types(it, struct_type.members) {
        struct_elem[i] = mk_tuple_select(source, i);
        i++;
      }

      for (unsigned k = 0; k < num_elems; k++)
        struct_elem_inv[(num_elems - 1) - k] = struct_elem[k];

      for (unsigned k = 0; k < num_elems; k++)
      {
        if (k == 1)
          struct_elem_inv[num_elems] = z3::to_expr(ctx, Z3_mk_concat(
            z3_ctx, struct_elem_inv[k - 1], struct_elem_inv[k]));
        else if (k > 1)
          struct_elem_inv[num_elems] = z3::to_expr(ctx, Z3_mk_concat(
            z3_ctx, struct_elem_inv[num_elems], struct_elem_inv[k]));
      }

      source = struct_elem_inv[num_elems];
    }

    unsigned int sort_sz =Z3_get_bv_sort_size(ctx, Z3_get_sort(ctx, source));
    if (sort_sz < upper) {
      // Extends past the end of this data item. Should be fixed in some other
      // dedicated feature branch, in the meantime stop Z3 from crashing
      z3::sort s = ctx.bv_sort(8);
      output = ctx.fresh_const(NULL, s);
    } else {
      output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, upper, lower, source));
    }
  }
}

void
z3_convt::convert_smt_expr(const byte_update2t &data, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  if (!is_constant_int2t(data.source_offset))
    throw new conv_error("byte_extract expects constant 2nd arg");

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  z3::expr tuple, value;
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
      output = mk_tuple_update(tuple, intref.constant_value.to_long(), value);
    else
      output = z3::to_expr(ctx, tuple);
  } else if (is_signedbv_type(data.source_value->type)) {
    if (int_encoding) {
      output = z3::to_expr(ctx, value);
      return;
    }

    width_op0 = data.source_value->type->get_width();

    if (width_op0 == 0)
      // XXXjmorse - can this ever happen now?
      throw new conv_error("failed to get width of byte_update operand");

    if (width_op0 > width_op2)
      output = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, (width_op0 - width_op2), value));
    else
      throw new conv_error("unsupported irep for conver_byte_update");
  } else {
    throw new conv_error("unsupported irep for conver_byte_update");
  }

}

void
z3_convt::convert_smt_expr(const with2t &with, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr operand0, operand1, operand2;
  z3::expr tuple, value;

  if (is_structure_type(with.type)) {
    unsigned int idx = 0;
    const struct_union_data &data_ref =
      dynamic_cast<const struct_union_data &>(*with.type);
    const std::vector<irep_idt> &names = data_ref.get_structure_member_names();

    convert_bv(with.source_value, tuple);
    convert_bv(with.update_value, value);

    const constant_string2t &str = to_constant_string2t(with.update_field);

    forall_names(it, names) {
      if (*it == str.value)
        break;
      idx++;
    }

    assert(idx != names.size() &&
           "Member name of with expr not found in struct/union type");

    output = mk_tuple_update(tuple, idx, value);

    // Update last-updated-field field if it's a union
    if (is_union_type(with.type)) {
      const union_type2t &unionref = to_union_type(with.type);
       unsigned int components_size = unionref.members.size();
       output = mk_tuple_update(output, components_size,
                                ctx.esbmc_int_val(idx));
    }
  } else if (is_array_type(with.type)) {

    convert_bv(with.source_value, operand0);
    convert_bv(with.update_field, operand1);
    convert_bv(with.update_value, operand2);

    output = z3::to_expr(ctx, Z3_mk_store(z3_ctx, operand0, operand1, operand2));
  } else {
    throw new conv_error("with applied to non-struct/union/array obj");
  }
}

void
z3_convt::convert_smt_expr(const member2t &member, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr struct_var;
  u_int j = 0;

  const struct_union_data &data_ref =
    dynamic_cast<const struct_union_data &>(*member.source_value->type);
  const std::vector<irep_idt> &member_names =
    data_ref.get_structure_member_names();

  forall_names(it, member_names) {
    if (*it == member.member.as_string())
      break;
    j++;
  }

  convert_bv(member.source_value, struct_var);

  if (is_union_type(member.source_value->type)) {
    union_varst::const_iterator cache_result;

    if (is_symbol2t(member.source_value)) {
      const symbol2t &sym = to_symbol2t(member.source_value);
      cache_result = union_vars.find(sym.get_symbol_name().c_str());
    } else {
      cache_result = union_vars.end();
    }

    if (cache_result != union_vars.end()) {
      const std::vector<type2tc> &members = data_ref.get_structure_members();

      const type2tc source_type = members[cache_result->idx];
      if (source_type == member.type) {
        // Type we're fetching from union matches expected type; just return it.
        output = mk_tuple_select(struct_var, cache_result->idx);
        return;
      }

      // Union field and expected type mismatch. Need to insert a cast.
      // Duplicate expr as we're changing it
      expr2tc memb2(new member2t(source_type, member.source_value, member.member));
      expr2tc cast(new typecast2t(member.type, memb2));
      convert_bv(cast, output);
      return;
    }
  }

  output = mk_tuple_select(struct_var, j);

#if 0
  // XXX jmorse - this turned up during merging some latest kinduction goo into
  // irep2, pre master merge. Not certain why it exists, appears to insert an
  // inverted cast from int to bool into certain members, when k-induction is
  // enabled.
  // Left here in case it turns out some tests are failing in the future.
  if (config.options.get_bool_option("k-induction"))
  {
    if (expr.type().is_bool() &&
        expr.op0().type().is_struct())
    {
      Z3_ast cond;
      unsigned width;
      const struct_typet &struct_type = to_struct_type(expr.op0().type());

      get_type_width(struct_type.components()[j].type(), width);

      if (struct_type.components()[j].type().id() == "signedbv")
        cond = Z3_mk_eq(z3_ctx, bv, convert_number(0, width, true));
      else if (struct_type.components()[j].type().id() == "unsignedbv")
        cond = Z3_mk_eq(z3_ctx, bv, convert_number(0, width, false));
      else if (struct_type.components()[j].type().id() == "bool")
        cond = Z3_mk_eq(z3_ctx, bv, Z3_mk_false(z3_ctx));
      else
      {
        std::cout << "we do not support `" << struct_type.components()[j].type().id() 
                  << "' in the state vector" << std::endl;
        assert(0);
      }

      bv = Z3_mk_ite(z3_ctx, cond, Z3_mk_false(z3_ctx), Z3_mk_true(z3_ctx));
#endif
}

void
z3_convt::convert_typecast_bool(const typecast2t &cast, z3::expr &output)
{

  if (is_bv_type(cast.from->type) ||
      is_pointer_type(cast.from->type)) {
    output = output != ctx.esbmc_int_val(0);
  } else {
    throw new conv_error("Unimplemented bool typecast");
  }
}

void
z3_convt::convert_typecast_fixedbv_nonint(const typecast2t &cast,
                                          z3::expr &output)
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
      output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, (from_width - 1), to_integer_bits, output));
    } else {
      assert(from_width < to_integer_bits);
      output = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_width), output));
    }

    output = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, output, ctx.esbmc_int_val(0, to_fraction_bits)));
  } else if (is_bool_type(cast.from->type)) {
    z3::expr zero, one;
    zero = ctx.esbmc_int_val(0, to_integer_bits);
    one = ctx.esbmc_int_val(1, to_integer_bits);
    output = z3::ite(output, one, zero);
    output = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, output, ctx.esbmc_int_val(0, to_fraction_bits)));
  } else if (is_fixedbv_type(cast.from->type)) {
    z3::expr magnitude, fraction;

    const fixedbv_type2t &from_fbvt = to_fixedbv_type(cast.from->type);

    unsigned from_fraction_bits = from_fbvt.width - from_fbvt.integer_bits;
    unsigned from_integer_bits = from_fbvt.integer_bits;
    unsigned from_width = from_fbvt.width;

    if (to_integer_bits <= from_integer_bits) {
      magnitude = z3::to_expr(ctx,
        Z3_mk_extract(z3_ctx, (from_fraction_bits + to_integer_bits - 1),
                      from_fraction_bits, output));
    } else   {
      assert(to_integer_bits > from_integer_bits);

      z3::expr ext = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, from_width - 1,
                                                     from_fraction_bits,
                                                     output));
      magnitude = z3::to_expr(ctx,
        Z3_mk_sign_ext(z3_ctx, (to_integer_bits - from_integer_bits), ext));
    }

    if (to_fraction_bits <= from_fraction_bits) {
      fraction = z3::to_expr(ctx,
        Z3_mk_extract(z3_ctx, (from_fraction_bits - 1),
                      from_fraction_bits - to_fraction_bits,
                      output));
    } else   {
      assert(to_fraction_bits > from_fraction_bits);

      z3::expr ext = z3::to_expr(ctx,
          Z3_mk_extract(z3_ctx, (from_fraction_bits - 1), 0, output));
      z3::expr zero =
        ctx.esbmc_int_val(0, to_fraction_bits - from_fraction_bits);

      fraction = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, ext, zero));
    }
    output = z3::to_expr(ctx, Z3_mk_concat(z3_ctx, magnitude, fraction));
  } else {
    throw new conv_error("unexpected typecast to fixedbv");
  }

  return;
}

void
z3_convt::convert_typecast_to_ints(const typecast2t &cast, z3::expr &output)
{
  unsigned to_width = cast.type->get_width();

  if (is_signedbv_type(cast.from->type) ||
      is_fixedbv_type(cast.from->type)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      if (int_encoding && is_signedbv_type(cast.from->type) &&
               is_fixedbv_type(cast.type))
	output = z3::to_expr(ctx, Z3_mk_int2real(z3_ctx, output));
      else if (int_encoding && is_fixedbv_type(cast.from->type) &&
               is_signedbv_type(cast.type))
	output = z3::to_expr(ctx, Z3_mk_real2int(z3_ctx, output));
      // XXXjmorse - there isn't a case here for if !int_encoding

    } else if (from_width < to_width)      {
      if (int_encoding &&
          ((is_fixedbv_type(cast.type) &&
            is_signedbv_type(cast.from->type))))
	output = z3::to_expr(ctx, Z3_mk_int2real(z3_ctx, output));
      else if (int_encoding)
	; // output = output
      else
	output = z3::to_expr(ctx, Z3_mk_sign_ext(z3_ctx, (to_width - from_width), output));
    } else if (from_width > to_width)     {
      if (int_encoding &&
          ((is_signedbv_type(cast.from->type) &&
            is_fixedbv_type(cast.type))))
	output = z3::to_expr(ctx, Z3_mk_int2real(z3_ctx, output));
      else if (int_encoding &&
               (is_fixedbv_type(cast.from->type) &&
                is_signedbv_type(cast.type)))
	output = z3::to_expr(ctx, Z3_mk_real2int(z3_ctx, output));
      else if (int_encoding)
	; // output = output
      else {
	if (!to_width) to_width = config.ansi_c.int_width;
	output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, (to_width - 1), 0, output));
      }
    }
  } else if (is_unsignedbv_type(cast.from->type)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      ; // output = output
    } else if (from_width < to_width)      {
      if (int_encoding)
	; // output = output
      else
	output = z3::to_expr(ctx, Z3_mk_zero_ext(z3_ctx, (to_width - from_width), output));
    } else if (from_width > to_width)     {
      if (int_encoding)
	; // output = output
      else
	output = z3::to_expr(ctx, Z3_mk_extract(z3_ctx, (to_width - 1), 0, output));
    }
  } else if (is_bool_type(cast.from->type)) {
    z3::expr zero, one;
    unsigned width = cast.type->get_width();

    if (is_bv_type(cast.type)) {
      zero = ctx.esbmc_int_val(0, width);
      one = ctx.esbmc_int_val(1, width);
    } else if (is_fixedbv_type(cast.type)) {
      zero = ctx.real_val(0);
      one = ctx.real_val(1);
    } else {
      throw new conv_error("Unexpected type in typecast of bool");
    }
    output = z3::ite(output, one, zero);
  } else   {
    throw new conv_error("Unexpected type in int/ptr typecast");
  }
}

void
z3_convt::convert_typecast_struct(const typecast2t &cast, z3::expr &output)
{
  const struct_type2t &struct_type_from = to_struct_type(cast.from->type);
  const struct_type2t &struct_type_to = to_struct_type(cast.type);

  z3::expr freshval;
  u_int i = 0, i2 = 0;

  std::vector<type2tc> new_members;
  std::vector<irep_idt> new_names;
  new_members.reserve(struct_type_to.members.size());
  new_names.reserve(struct_type_to.members.size());

  i = 0;
  forall_types(it, struct_type_to.members) {
    if (!base_type_eq(struct_type_from.members[i], *it, ns))
      throw new conv_error("Incompatible struct in cast-to-struct");

    i++;
  }

  z3::sort sort;
  convert_type(cast.type, sort);

  freshval = ctx.fresh_const(NULL, sort);

  i2 = 0;
  forall_types(it, struct_type_to.members) {
    z3::expr formula;
    formula = mk_tuple_select(freshval, i2) == mk_tuple_select(output, i2);
    assert_formula(formula);
    i2++;
  }

  output = freshval;
  return;
}

void
z3_convt::convert_typecast_to_ptr(const typecast2t &cast, z3::expr &output)
{

  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if (is_pointer_type(cast.from->type)) {
    // output is already plain-converted.
    return;
  }

  // Unpleasentness; we don't know what pointer this integer is going to
  // correspond to, and there's no way of telling statically, so we have
  // to enumerate all pointers it could point at. IE, all of them. Which
  // is expensive, but here we are.

  // First cast it to an unsignedbv
  z3::expr target;
  type2tc int_type(new unsignedbv_type2t(config.ansi_c.int_width));
  expr2tc cast_to_unsigned(new typecast2t(int_type, cast.from));
  convert_bv(cast_to_unsigned, target);

  // Construct array for all possible object outcomes
  z3::expr is_in_range[addr_space_data.back().size()];
  z3::expr obj_ids[addr_space_data.back().size()];
  z3::expr obj_starts[addr_space_data.back().size()];

  std::map<unsigned,unsigned>::const_iterator it;
  unsigned int i;
  for (it = addr_space_data.back().begin(), i = 0;
       it != addr_space_data.back().end(); it++, i++)
  {
    unsigned id = it->first;
    obj_ids[i] = ctx.esbmc_int_val(id);
    z3::expr start = ctx.constant(
                                 ("__ESBMC_ptr_obj_start_" + itos(id)).c_str(),
                                 ctx.esbmc_int_sort());
    z3::expr end = ctx.constant(
                                 ("__ESBMC_ptr_obj_end_" + itos(id)).c_str(),
                                 ctx.esbmc_int_sort());
    obj_starts[i] = start;

    is_in_range[i] = mk_ge(target, start, true) && mk_le(target, end, true);
  }

  // Generate a big ITE chain, selecing a particular pointer offset. A
  // significant question is what happens when it's neither; in which case I
  // suggest the ptr becomes invalid_object. However, this needs frontend
  // support to check for invalid_object after all dereferences XXXjmorse.

  // So, what's the default value going to be if it doesn't match any existing
  // pointers? Answer, it's going to be the invalid object identifier, but with
  // an offset that calculates to the integer address of this object.
  // That's so that we can store an invalid pointer in a pointer type, that
  // eventually can be converted back via some mechanism to a valid pointer.
  z3::expr args[2];
  args[0] = ctx.esbmc_int_val(pointer_logic.back().get_invalid_object());

  // Calculate ptr offset - target minus start of invalid range, ie 1
  args[1] = target - ctx.esbmc_int_val(1);

  z3::expr prev_in_chain = pointer_decl(args[0], args[1]);

  // Now that big ite chain,
  for (i = 0; i < addr_space_data.back().size(); i++) {
    args[0] = obj_ids[i];

    // Calculate ptr offset were it this
    args[1] = target - obj_starts[i];

    z3::expr selected_tuple = pointer_decl(args[0], args[1]);

    prev_in_chain = z3::ite(is_in_range[i], selected_tuple, prev_in_chain);
  }

  // Finally, we're now at the point where prev_in_chain represents a pointer
  // object. Hurrah.
  output = z3::to_expr(ctx, prev_in_chain);
}

void
z3_convt::convert_typecast_from_ptr(const typecast2t &cast, z3::expr &output)
{
  type2tc int_type(new unsignedbv_type2t(config.ansi_c.int_width));

  // The plan: index the object id -> address-space array and pick out the
  // start address, then add it to any additional pointer offset.

  // Generate type of address space array
  std::vector<type2tc> members;
  std::vector<irep_idt> names;
  type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
  members.push_back(inttype);
  members.push_back(inttype);
  names.push_back(irep_idt("start"));
  names.push_back(irep_idt("end"));
  type2tc strct(new struct_type2t(members, names,
                irep_idt("addr_space_tuple")));
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
  convert_bv(new_cast, output);
}

void
z3_convt::convert_smt_expr(const typecast2t &cast, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  convert_bv(cast.from, output);

  if (is_pointer_type(cast.type)) {
    convert_typecast_to_ptr(cast, output);
  } else if (is_pointer_type(cast.from->type)) {
    convert_typecast_from_ptr(cast, output);
  } else if (is_bool_type(cast.type)) {
    convert_typecast_bool(cast, output);
  } else if (is_fixedbv_type(cast.type) && !int_encoding)      {
    convert_typecast_fixedbv_nonint(cast, output);
  } else if (is_bv_type(cast.type) ||
             is_fixedbv_type(cast.type) ||
             is_pointer_type(cast.type)) {
    convert_typecast_to_ints(cast, output);
  } else if (is_struct_type(cast.type))     {
    convert_typecast_struct(cast, output);
  } else if (is_union_type(cast.type)) {
    if (base_type_eq(cast.type, cast.from->type, namespacet(contextt())))
      return; // No additional conversion required
    else
      throw new conv_error("Can't typecast between unions");
  } else {
    // XXXjmorse -- what about all other types, eh?
    throw new conv_error("Typecast for unexpected type");
  }
}

void
z3_convt::convert_smt_expr(const index2t &index, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  z3::expr source, idx;

  convert_bv(index.source_value, source);
  convert_bv(index.index, idx);

  // XXXjmorse - consider situation where a pointer is indexed. Should it
  // give the address of ptroffset + (typesize * index)?
  output = select(source, idx);
}

void
z3_convt::convert_smt_expr(const zero_string2t &zstr, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  // XXXjmorse - this method appears to just return a free variable. Surely
  // it should be selecting the zero_string field out of the referenced
  // string?
  z3::sort array_type;

  convert_type(zstr.type, array_type);

  output = z3::to_expr(ctx, ctx.constant("zero_string", array_type));
}

void
z3_convt::convert_smt_expr(const zero_length_string2t &s, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr operand;

  convert_bv(s.string, operand);
  output = mk_tuple_select(operand, 0);
}

void
z3_convt::convert_smt_expr(const isnan2t &isnan, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);

  if (is_fixedbv_type(isnan.value->type)) {
    z3::expr op0;
    unsigned width = isnan.value->type->get_width();

    convert_bv(isnan.value, op0);

    z3::expr t = ctx.bool_val(true);
    z3::expr f = ctx.bool_val(false);
    if (int_encoding) {
      z3::expr zero = ctx.esbmc_int_val(0, width);
      z3::expr r2int = z3::to_expr(ctx, Z3_mk_real2int(z3_ctx, op0));
      z3::expr ge = mk_ge(r2int, zero, true); // sign unimportant in int mode

      output = z3::to_expr(ctx, Z3_mk_ite(z3_ctx, ge, t, f));
    } else {
      z3::expr zero = ctx.esbmc_int_val(0, width);
      z3::expr ge = mk_ge(op0, zero, false); // In original, always signed ge?
      output = z3::to_expr(ctx, Z3_mk_ite(z3_ctx, ge, t, f));
    }
  } else {
    throw new conv_error("isnan with unsupported operand type");
  }
}

void
z3_convt::convert_smt_expr(const overflow2t &overflow, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr result[2], operand[2];
  unsigned width_op0, width_op1;

  // XXX jmorse - we can't tell whether or not we're supposed to be treating
  // the _result_ as being a signedbv or an unsignedbv, because we only have
  // operands. Ideally, this needs to be encoded somewhere.
  // Specifically, when irep2 conversion reaches code creation, we should
  // encode the resulting type in the overflow operands type. Right now it's
  // inferred.
  Z3_bool is_signed = Z3_L_FALSE;

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
    call1 = workaround_Z3_mk_bvadd_no_overflow;
    call2 = workaround_Z3_mk_bvadd_no_underflow;
    if (is_signedbv_type(to_add2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_add2t(overflow.operand).side_2->type))
      is_signed = Z3_L_TRUE;
  } else if (is_sub2t(overflow.operand)) {
    convert_bv(to_sub2t(overflow.operand).side_1, operand[0]);
    convert_bv(to_sub2t(overflow.operand).side_2, operand[1]);
    width_op0 = to_sub2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_sub2t(overflow.operand).side_2->type->get_width();
    call1 = workaround_Z3_mk_bvsub_no_underflow;
    call2 = workaround_Z3_mk_bvsub_no_overflow;
    if (is_signedbv_type(to_sub2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_sub2t(overflow.operand).side_2->type))
      is_signed = Z3_L_TRUE;
  } else if (is_mul2t(overflow.operand)) {
    convert_bv(to_mul2t(overflow.operand).side_1, operand[0]);
    convert_bv(to_mul2t(overflow.operand).side_2, operand[1]);
    width_op0 = to_mul2t(overflow.operand).side_1->type->get_width();
    width_op1 = to_mul2t(overflow.operand).side_2->type->get_width();
    // XXX jmorse - no reference counting workaround for this; disassembling
    // these Z3 routines show that they've been touched by reference count
    // switchover, and so are likely actually reference counting correctly.
    call1 = Z3_mk_bvmul_no_overflow;
    call2 = Z3_mk_bvmul_no_underflow;
    if (is_signedbv_type(to_mul2t(overflow.operand).side_1->type) ||
        is_signedbv_type(to_mul2t(overflow.operand).side_2->type))
      is_signed = Z3_L_TRUE;
  } else {
    std::cerr << "Overflow operation with invalid operand";
    abort();
  }

  // XXX jmorse - int2bv trainwreck.
  if (int_encoding) {
    operand[0] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op0, operand[0]));
    operand[1] = z3::to_expr(ctx, Z3_mk_int2bv(z3_ctx, width_op1, operand[1]));
  }

  result[0] = z3::to_expr(ctx, call1(z3_ctx, operand[0], operand[1], is_signed));
  result[1] = z3::to_expr(ctx, call2(z3_ctx, operand[0], operand[1]));
  output = !(result[0] && result[1]);
}

void
z3_convt::convert_smt_expr(const overflow_cast2t &ocast, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  uint64_t result;
  u_int width;

  width = ocast.operand->type->get_width();

  if (ocast.bits >= width || ocast.bits == 0)
    throw new conv_error("overflow-typecast got wrong number of bits");

  assert(ocast.bits <= 32 && ocast.bits != 0);
  result = 1 << ocast.bits;

  expr2tc oper = ocast.operand;

  // Cast fixedbv to its integer form.
  if (is_fixedbv_type(ocast.operand->type)) {
    const fixedbv_type2t &fbvt = to_fixedbv_type(ocast.operand->type);
    type2tc signedbv(new signedbv_type2t(fbvt.integer_bits));
    oper = expr2tc(new typecast2t(signedbv, oper));
  }

  expr2tc lessthan, greaterthan;
  if (is_signedbv_type(ocast.operand->type) ||
      is_fixedbv_type(ocast.operand->type)) {
    // Produce some useful constants
    unsigned int nums_width = (is_signedbv_type(ocast.operand->type))
                               ? width : width / 2;
    type2tc signedbv(new signedbv_type2t(nums_width));
    expr2tc result_val(new constant_int2t(signedbv, BigInt(result / 2)));
    expr2tc two(new constant_int2t(signedbv, BigInt(2)));
    expr2tc minus_one(new constant_int2t(signedbv, BigInt(-1)));

    // Now produce numbers that bracket the selected bitwidth. So for 16 bis
    // we would generate 2^15-1 and -2^15
    expr2tc upper(new sub2t(signedbv, result_val, minus_one));
    expr2tc lower(new mul2t(signedbv, result_val, minus_one));

    // Ensure operand lies between these braces
    lessthan = expr2tc(new lessthan2t(oper, upper));
    greaterthan = expr2tc(new greaterthan2t(oper, lower));
  } else if (is_unsignedbv_type(ocast.operand->type)) {
    // Create zero and 2^bitwidth,
    type2tc unsignedbv(new unsignedbv_type2t(width));

    expr2tc zero(new constant_int2t(unsignedbv, BigInt(0)));
    expr2tc the_width(new constant_int2t(unsignedbv, BigInt(result)));

    // Ensure operand lies between those numbers.
    lessthan = expr2tc(new lessthan2t(oper, the_width));
    greaterthan = expr2tc(new greaterthanequal2t(oper, zero));
  }

  z3::expr ops[2];
  convert_bv(lessthan, ops[0]);
  convert_bv(greaterthan, ops[1]);

  output = !(ops[0] && ops[1]);
}

void
z3_convt::convert_smt_expr(const overflow_neg2t &neg, void *_bv)
{
  z3::expr &output = cast_to_z3(_bv);
  z3::expr operand;
  unsigned width;

  convert_bv(neg.operand, operand);

  // XXX jmorse - clearly wrong. Neg of pointer?
  if (is_pointer_type(neg.operand->type))
    operand = mk_tuple_select(operand, 1);

  width = neg.operand->type->get_width();

  // XXX jmorse - int2bv trainwreck
  if (int_encoding)
    operand = to_expr(ctx, Z3_mk_int2bv(z3_ctx, width, operand));

  z3::expr no_over = z3::to_expr(ctx,
                           workaround_Z3_mk_bvneg_no_overflow(z3_ctx, operand));
  output = z3::to_expr(ctx, Z3_mk_not(z3_ctx, no_over));
}

void
z3_convt::convert_pointer_arith(expr2t::expr_ids id, const expr2tc &side1,
                                const expr2tc &side2,
                                const type2tc &type, z3::expr &output)
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
      convert_bv(cast, output);
      break;
      }
    case 5:
    case 6:
      {
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      // Actually perform some pointer arith
      const pointer_type2t &ptr_type = to_pointer_type(ptr_op->type);
      typet followed_type_old = ns.follow(migrate_type_back(ptr_type.subtype));
      type2tc followed_type;
      migrate_type(followed_type_old, followed_type);
      mp_integer type_size = pointer_offset_size(*followed_type);

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
      convert_bv(newexpr, output);

      // That calculated the offset; update field in pointer.
      z3::expr the_ptr;
      convert_bv(ptr_op, the_ptr);
      output = mk_tuple_update(the_ptr, 1, output);

      break;
      }
  }
}

void
z3_convt::convert_bv(const expr2tc &expr, z3::expr &val)
{

  bv_cachet::const_iterator cache_result = bv_cache.find(expr);
  if (cache_result != bv_cache.end()) {
    val = cache_result->output;
    return;
  }

  expr->convert_smt(*this, reinterpret_cast<void*>(&val));

  // insert into cache
  struct bv_cache_entryt cacheentry = { expr, val, level_ctx };
  bv_cache.insert(cacheentry);
  return;
}

void
z3_convt::convert_type(const type2tc &type, z3::sort &outtype)
{

  sort_cachet::const_iterator cache_result = sort_cache.find(type);
  if (cache_result != sort_cache.end()) {
    outtype = z3::to_sort(ctx, cache_result->second);
    return;
  }

  type->convert_smt_type(*this, reinterpret_cast<void*>(&outtype));

  // insert into cache
  sort_cache.insert(std::pair<const type2tc, z3::sort>(type, outtype));
  return;
}

literalt
z3_convt::convert_expr(const expr2tc &expr)
{
  literalt l = new_variable();
  z3::expr formula, constraint;

  expr2tc new_expr;

  try {
    convert_bv(expr, constraint);
  } catch (std::string *e) {
    std::cerr << "Failed to convert an expression" << std::endl;
    ignoring(expr);
    return l;
  } catch (conv_error *e) {
    std::cerr << e->to_string() << std::endl;
    ignoring(expr);
    return l;
  }

  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, constraint));

  // While we have a literal, don't assert that it's true, only the link
  // between the formula and the literal. Otherwise, we risk asserting that a
  // formula within a assertion-statement is true or false.
  assert_formula(formula);

  return l;
}

void
z3_convt::convert_identifier_pointer(const expr2tc &expr, std::string symbol,
                                     z3::expr &output)
{
  std::string cte, identifier;
  unsigned int obj_num;
  bool got_obj_num = false;

  if (is_symbol2t(expr)) {
    const symbol2t &sym = to_symbol2t(expr);
    if (sym.thename == "NULL" || sym.thename == "0") {
      obj_num = pointer_logic.back().get_null_object();
      got_obj_num = true;
    }
  }

  if (!got_obj_num)
    // add object won't duplicate objs for identical exprs (it's a map)
    obj_num = pointer_logic.back().add_object(expr);

  output = z3::to_expr(ctx, ctx.constant(symbol.c_str(), pointer_sort));

  // If this object hasn't yet been put in the address space record, we need to
  // assert that the symbol has the object ID we've allocated, and then fill out
  // the address space record.
  if (addr_space_data.back().find(obj_num) == addr_space_data.back().end()) {

    z3::expr ptr_val = pointer_decl(ctx.esbmc_int_val(obj_num),
                                       ctx.esbmc_int_val(0));

    z3::expr constraint = output == ptr_val;
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
    total_mem_space.back() +=
      pointer_offset_size(*expr->type.get()).to_long() + 1;

    // Assert that start + offs == end
    z3::expr offs_eq;
    convert_bv(endisequal, offs_eq);
    assert_formula(offs_eq);

    // Even better, if we're operating in bitvector mode, it's possible that
    // Z3 will try to be clever and arrange the pointer range to cross the end
    // of the address space (ie, wrap around). So, also assert that end > start
    expr2tc wraparound(new greaterthan2t(end_sym, start_sym));
    z3::expr wraparound_eq;
    convert_bv(wraparound, wraparound_eq);
    assert_formula(wraparound_eq);

    // Generate address space layout constraints.
    finalize_pointer_chain(obj_num);

    addr_space_data.back()[obj_num] =
          pointer_offset_size(*expr->type.get()).to_long() + 1;

    z3::expr start_ast, end_ast;
    convert_bv(start_sym, start_ast);
    convert_bv(end_sym, end_ast);

    // Actually store into array
    z3::expr range_tuple = ctx.constant(
                       ("__ESBMC_ptr_addr_range_" + itos(obj_num)).c_str(),
                       addr_space_tuple_sort);
    z3::expr init_val =
      addr_space_tuple_decl.make_tuple("", &start_ast, &end_ast, NULL);
    z3::expr eq = range_tuple == init_val;
    assert_formula(eq);

    // Update array
    bump_addrspace_array(obj_num, range_tuple);

    // Finally, ensure that the array storing whether this pointer is dynamic,
    // is initialized for this ptr to false. That way, only pointers created
    // through malloc will be marked dynamic.

    type2tc arrtype(new array_type2t(type2tc(new bool_type2t()),
                                     expr2tc((expr2t*)NULL), true));
    expr2tc allocarr(new symbol2t(arrtype, dyn_info_arr_name));
    z3::expr allocarray;
    convert_bv(allocarr, allocarray);

    z3::expr idxnum = ctx.esbmc_int_val(obj_num);
    z3::expr select = z3::select(allocarray, idxnum);
    z3::expr isfalse = ctx.bool_val(false) == select;
    assert_formula(isfalse);
  }

  DEBUGLOC;
}

literalt
z3_convt::land(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  z3::expr args[size];
  Z3_ast args_ast[size];
  z3::expr result, formula;

  for (unsigned int i = 0; i < bv.size(); i++) {
    args[i] = z3_literal(bv[i]);
    args_ast[i] = args[i];
  }

  result = to_expr(ctx, Z3_mk_and(z3_ctx, bv.size(), args_ast));
  z3::expr thelit = z3_literal(l);
  formula = to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
  assert_formula(formula);

  return l;
}

literalt
z3_convt::lor(const bvt &bv)
{

  literalt l = new_variable();
  uint size = bv.size();
  z3::expr args[size];
  Z3_ast args_ast[size];
  z3::expr result, formula;

  for (unsigned int i = 0; i < bv.size(); i++) {
    args[i] = z3_literal(bv[i]);
    args_ast[i] = args[i];
  }

  result = z3::to_expr(ctx, Z3_mk_or(z3_ctx, bv.size(), args_ast));

  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
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
  z3::expr result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = operand[0] && operand[1];
  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
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
  z3::expr result, operand[2], formula;

  operand[0] = z3_literal(a);
  operand[1] = z3_literal(b);
  result = operand[0] || operand[1];
  z3::expr thelit = z3_literal(l);
  formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, result));
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

  l.set(no_variables, false);

  set_no_variables(no_variables + 1);

  return l;
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

    assert(l.var_no() < no_variables);

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

  z3::expr lor_var, args[new_bv.size()];
  Z3_ast args_ast[new_bv.size()];
  unsigned int i = 0;

  for (bvt::const_iterator it = new_bv.begin(); it != new_bv.end(); it++, i++) {
    args[i] = z3_literal(*it);
    args_ast[i] = args[i];
  }

  if (i > 1) {
    lor_var = z3::expr(ctx, Z3_mk_or(z3_ctx, i, args_ast));
    assert_formula(lor_var);
  } else   {
    assert_formula(args[0]);
  }
}

z3::expr
z3_convt::z3_literal(literalt l)
{

  z3::expr literal_l;
  std::string literal_s;

  if (l == const_literal(false))
    return ctx.bool_val(false);
  else if (l == const_literal(true))
    return ctx.bool_val(true);

  literal_s = "l" + i2string(l.var_no());
  literal_l = ctx.constant(literal_s.c_str(), ctx.bool_sort());

  if (l.sign()) {
    return !literal_l;
  }

  return literal_l;
}

tvt
z3_convt::l_get(literalt a)
{
  tvt result = tvt(tvt::TV_ASSUME);
  std::string literal;

  if (a.is_true()) {
    return tvt(true);
  } else if (a.is_false())    {
    return tvt(false);
  }

  expr2tc sym(new symbol2t(type_pool.get_bool(),
                           irep_idt("l" + i2string(a.var_no()))));
  expr2tc res = get(sym);

  if (!is_nil_expr(res) && is_constant_bool2t(res)) {
    result = (to_constant_bool2t(res).is_true())
             ? tvt(tvt::TV_TRUE) : tvt(tvt::TV_FALSE);
  } else {
    result = tvt(tvt::TV_UNKNOWN);
  }

  if (a.sign())
    result = !result;

  return result;
}

void
z3_convt::assert_lit(const literalt &l)
{
  z3::expr thelit = z3_literal(l);
  solver.add(thelit);
  assumpt.push_back(thelit);
}

void
z3_convt::assert_formula(const z3::expr &ast)
{

  // If we're not going to be using the assumptions (ie, for unwidening and for
  // smtlib) then just assert the fact to be true.
  if (!assumpt_mode) {
    solver.add(ast);
    return;
  }

  literalt l = new_variable();
  z3::expr thelit = z3_literal(l);
  z3::expr formula = z3::to_expr(ctx, Z3_mk_iff(z3_ctx, thelit, ast));
  solver.add(formula);

  // jmorse - delete when smtlib printer exists and works
#if 0
  if (smtlib)
    assumpt.push_back(ast);
  else
#endif
    assumpt.push_back(z3_literal(l));

  return;
}

z3::expr
z3_convt::mk_tuple_update(const z3::expr &t, unsigned i, const z3::expr &newval)
{
  z3::sort ty;
  unsigned num_fields, j;

  ty = t.get_sort();

  if (!ty.is_datatype()) {
    std::cerr << "argument must be a tuple";
    abort();
  }

  num_fields = Z3_get_tuple_sort_num_fields(ctx, ty);

  if (i >= num_fields) {
    std::cerr << "invalid tuple update, index is too big";
    abort();
  }

  z3::expr new_fields[num_fields];
  for (j = 0; j < num_fields; j++) {
    if (i == j) {
      /* use new_val at position i */
      new_fields[j] = newval;
    } else   {
      /* use field j of t */
      z3::func_decl proj_decl =
        z3::to_func_decl(ctx, Z3_get_tuple_sort_field_decl(ctx, ty, j));
      new_fields[j] = proj_decl(t);
    }
  }

  z3::func_decl mk_tuple_decl =
    z3::to_func_decl(ctx, Z3_get_tuple_sort_mk_decl(ctx, ty));

  return mk_tuple_decl.make_tuple_from_array(num_fields, new_fields);
}

z3::expr
z3_convt::mk_tuple_select(const z3::expr &t, unsigned i)
{
  z3::sort ty;
  unsigned num_fields;

  ty = t.get_sort();

  if (!ty.is_datatype()) {
    throw new z3_convt::conv_error("argument must be a tuple");
  }

  num_fields = Z3_get_tuple_sort_num_fields(ctx, ty);

  if (i >= num_fields) {
    throw new z3_convt::conv_error("invalid tuple select, index is too big");
  }

  z3::func_decl proj_decl =
    z3::to_func_decl(ctx, Z3_get_tuple_sort_field_decl(ctx, ty, i));
  return proj_decl(t);
}

// SMT-abstraction migration routines.

smt_ast *
z3_convt::mk_func_app(const smt_sort *s __attribute__((unused)), smt_func_kind k __attribute__((unused)), smt_ast **args __attribute__((unused)), unsigned int numargs __attribute__((unused)))
{
  assert(0);
}

smt_sort *
z3_convt::mk_sort(const smt_sort_kind k __attribute__((unused)), ...)
{
  assert(0);
}

// Gigantic hack, implement a method in z3::ast, so that we can call from gdb
namespace z3 {
  void ast::dump(void) const {
    std::cout << Z3_ast_to_string(ctx(), m_ast) << std::endl;
    std::cout << "sort is " << Z3_sort_to_string(ctx(), Z3_get_sort(ctx(), m_ast)) << std::endl;
  }
};

// Caution: these workaround functions are a derived work from disassembling
// Z3 and adding reference counting to them. Technically they're derivative
// works that the microsoft research license demands be licensed under the
// microsoft research license, which puts us in a tricky copyright situation.
//
// Section 50C of the copyright act states that I can fix their software if
// necessary, but says nothing of distribution, and says that I can't ignore
// terms in MS' license restricting its lawful use.
//
// So, don't distribute this.
#ifdef NDEBUG
#error Don't distribute/release shadily licensed MS workaround code. And don't delete this error without asking jmorse.
#endif

Z3_ast
workaround_Z3_mk_bvadd_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                   Z3_bool is_signed)
{

  if (is_signed == Z3_L_TRUE) {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    Z3_ast zero = Z3_mk_int(ctx, 0, s);
    Z3_inc_ref(ctx, zero);
    Z3_ast add = Z3_mk_bvadd(ctx, a1, a2);
    Z3_inc_ref(ctx, add);
    Z3_ast lt1 = Z3_mk_bvslt(ctx, zero, a1);
    Z3_inc_ref(ctx, lt1);
    Z3_ast lt2 = Z3_mk_bvslt(ctx, zero, a2);
    Z3_inc_ref(ctx, lt2);
    Z3_ast args[2] = { lt1, lt2 };
    Z3_ast theand = Z3_mk_and(ctx, 2, args);
    Z3_inc_ref(ctx, theand);
    Z3_ast lt3 = Z3_mk_bvslt(ctx, zero, add);
    Z3_inc_ref(ctx, lt3);
    Z3_ast imp = Z3_mk_implies(ctx, theand, lt3);
    Z3_dec_ref(ctx, lt3);
    Z3_dec_ref(ctx, theand);
    Z3_dec_ref(ctx, lt2);
    Z3_dec_ref(ctx, lt1);
    Z3_dec_ref(ctx, add);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return imp;
  } else {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    unsigned int sort_size = Z3_get_bv_sort_size(ctx, s);
    Z3_ast ext1 = Z3_mk_zero_ext(ctx, 1, a1);
    Z3_inc_ref(ctx, ext1);
    Z3_ast ext2 = Z3_mk_zero_ext(ctx, 1, a2);
    Z3_inc_ref(ctx, ext2);
    Z3_ast add = Z3_mk_bvadd(ctx, ext1, ext2);
    Z3_inc_ref(ctx, add);
    Z3_sort s2 = Z3_mk_bv_sort(ctx, 1);
    Z3_inc_ref(ctx, (Z3_ast)s2);
    Z3_ast zero = Z3_mk_int(ctx, 0, s2);
    Z3_inc_ref(ctx, zero);
    Z3_ast ext = Z3_mk_extract(ctx, sort_size, sort_size, add);
    Z3_inc_ref(ctx, ext);
    Z3_ast eq = Z3_mk_eq(ctx, ext, zero);
    Z3_dec_ref(ctx, ext);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s2);
    Z3_dec_ref(ctx, add);
    Z3_dec_ref(ctx, ext2);
    Z3_dec_ref(ctx, ext1);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return eq;
  }
}

Z3_ast
workaround_Z3_mk_bvadd_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2)
{
  Z3_sort s = Z3_get_sort(ctx, a1);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast zero = Z3_mk_int(ctx, 0, s);
  Z3_inc_ref(ctx, zero);
  Z3_ast add = Z3_mk_bvadd(ctx, a1, a2);
  Z3_inc_ref(ctx, add);
  Z3_ast lt1 = Z3_mk_bvslt(ctx, a1, zero);
  Z3_inc_ref(ctx, lt1);
  Z3_ast lt2 = Z3_mk_bvslt(ctx, a2, zero);
  Z3_inc_ref(ctx, lt2);
  Z3_ast args[2] = { lt1, lt2 };
  Z3_ast theand = Z3_mk_and(ctx, 2, args);
  Z3_inc_ref(ctx, theand);
  Z3_ast lt3 = Z3_mk_bvslt(ctx, add, zero);
  Z3_inc_ref(ctx, lt3);
  Z3_ast imp = Z3_mk_implies(ctx, theand, lt3);
  Z3_dec_ref(ctx, lt3);
  Z3_dec_ref(ctx, theand);
  Z3_dec_ref(ctx, lt2);
  Z3_dec_ref(ctx, lt1);
  Z3_dec_ref(ctx, add);
  Z3_dec_ref(ctx, zero);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return imp;
}

Z3_ast
workaround_Z3_mk_bvsub_no_underflow(Z3_context ctx, Z3_ast a1, Z3_ast a2,
                                    Z3_bool is_signed)
{

  if (is_signed == Z3_L_TRUE) {
    Z3_sort s = Z3_get_sort(ctx, a1);
    Z3_inc_ref(ctx, (Z3_ast)s);
    Z3_ast zero = Z3_mk_int(ctx, 0, s);
    Z3_inc_ref(ctx, zero);
    Z3_ast neg = Z3_mk_bvneg(ctx, a2);
    Z3_inc_ref(ctx, neg);
    Z3_ast no_under = workaround_Z3_mk_bvadd_no_underflow(ctx, a1, neg);
    Z3_inc_ref(ctx, no_under);
    Z3_ast lt1 = Z3_mk_bvslt(ctx, zero, a2);
    Z3_inc_ref(ctx, lt1);
    Z3_ast imp = Z3_mk_implies(ctx, lt1, no_under);
    Z3_dec_ref(ctx, lt1);
    Z3_dec_ref(ctx, no_under);
    Z3_dec_ref(ctx, neg);
    Z3_dec_ref(ctx, zero);
    Z3_dec_ref(ctx, (Z3_ast)s);
    return imp;
  } else {
    return Z3_mk_bvule(ctx, a2, a1);
  }
}

extern "C" Z3_ast Z3_mk_bvsmin(Z3_context, Z3_sort);

Z3_ast
workaround_Z3_mk_bvsub_no_overflow(Z3_context ctx, Z3_ast a1, Z3_ast a2)
{

  Z3_sort s = Z3_get_sort(ctx, a2);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast neg = Z3_mk_bvneg(ctx, a2);
  Z3_inc_ref(ctx, neg);
//  Z3_ast min = Z3_mk_bvsmin(ctx, s);
//  Z3_inc_ref(ctx, min);
  Z3_ast min;
  {
    unsigned int width = Z3_get_bv_sort_size(ctx, s);
    Z3_ast sz = Z3_mk_int64(ctx, width - 1, s);
    Z3_inc_ref(ctx, sz);
    Z3_ast one = Z3_mk_int64(ctx, 1, s);
    Z3_inc_ref(ctx, one);
    Z3_ast msb = Z3_mk_bvshl(ctx, one, sz);
    Z3_inc_ref(ctx, msb);
    min = msb;
    Z3_dec_ref(ctx, one);
    Z3_dec_ref(ctx, sz);
  }
  Z3_ast no_over = workaround_Z3_mk_bvadd_no_overflow(ctx, a1, neg, 1);
  Z3_inc_ref(ctx, no_over);
  Z3_ast zero = Z3_mk_int(ctx, 0, s);
  Z3_inc_ref(ctx, zero);
  Z3_ast lt = Z3_mk_bvslt(ctx, a1, zero);
  Z3_inc_ref(ctx, lt);
  Z3_ast eq = Z3_mk_eq(ctx, a2, min);
  Z3_inc_ref(ctx, eq);
  Z3_ast ite = Z3_mk_ite(ctx, eq, lt, no_over);
  Z3_dec_ref(ctx, eq);
  Z3_dec_ref(ctx, lt);
  Z3_dec_ref(ctx, zero);
  Z3_dec_ref(ctx, no_over);
  Z3_dec_ref(ctx, min);
  Z3_dec_ref(ctx, neg);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return ite;
}

Z3_ast
workaround_Z3_mk_bvneg_no_overflow(Z3_context ctx, Z3_ast a)
{

  Z3_sort s = Z3_get_sort(ctx, a);
  Z3_inc_ref(ctx, (Z3_ast)s);
  Z3_ast min;
  {
    unsigned int width = Z3_get_bv_sort_size(ctx, s);
    Z3_ast sz = Z3_mk_int64(ctx, width - 1, s);
    Z3_inc_ref(ctx, sz);
    Z3_ast one = Z3_mk_int64(ctx, 1, s);
    Z3_inc_ref(ctx, one);
    Z3_ast msb = Z3_mk_bvshl(ctx, one, sz);
    Z3_inc_ref(ctx, msb);
    min = msb;
    Z3_dec_ref(ctx, one);
    Z3_dec_ref(ctx, sz);
  }
  Z3_ast eq = Z3_mk_eq(ctx, a, min);
  Z3_inc_ref(ctx, eq);
  Z3_ast thenot = Z3_mk_not(ctx, eq);
  Z3_dec_ref(ctx, eq);
  Z3_dec_ref(ctx, min);
  Z3_dec_ref(ctx, (Z3_ast)s);
  return thenot;
}
