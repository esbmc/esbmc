#include <algorithm>
#include <sstream>
#include <solvers/smt/smt_conv.h>
#include <util/message/format.h>
#include <util/type_byte_size.h>

/** @file smt_memspace.cpp
 *  Modelling the memory address space of C isn't something that is handled
 *  during any of the higher levels of ESBMC; it's instead left until the
 *  conversion to SMT to be handled.
 *
 *  The substance of what's done in this file orientates around the correct
 *  manipulations of anything in an expression that has a pointer type. This is
 *  then complicated by the C requirement that all pointers have some kind of
 *  integer representation (i.e., an address) that fits in the machine word.
 *  Furthermore, we have to be able to:
 *    * Cast pointers to and from these two representations.
 *    * Compare pointers, add pointers, and subtract pointers.
 *    * Identify pointers, for the 'same-object' test.
 *    * Take the address of expressions and produce a pointer.
 *    * Allow the integer representation of a pointer to be anywhere in the
 *      C memory address space, but for memory allocations to not overlap.
 *
 *  All of this is quite difficult, but do-able. We could instead just use an
 *  integer as the representation of any pointers, but that quickly becomes
 *  inefficient for some of the operations above. The substance of the solution
 *  is two things: firstly, we define a representation of a pointer type that
 *  is convenient for our operations. Secondly, we record enough information to
 *  map a pointer to its integer representation, and back again.
 */

smt_astt smt_convt::convert_ptr_cmp(
  const expr2tc &side1,
  const expr2tc &side2,
  const expr2tc &templ_expr)
{
  // Special handling for pointer comparisons (both ops are pointers; otherwise
  // it's obviously broken).
  assert(is_pointer_type(side1));
  assert(is_pointer_type(side2));
  assert(dynamic_cast<const relation_data *>(templ_expr.get()));

  /* Compare just the offsets. This is compatible with both, C and CHERI-C,
   * because we already asserted that they point to the same object (unless
   * --no-pointer-relation-check was specified, in which case the user opted
   * out of sanity anyway). */

  /* Create a copy of the expression and replace both sides with the respective
   * typecasted-to-unsigned versions of the offsets. The unsigned comparison is
   * required because objects could be larger than half the address space, in
   * which case offsets could flip sign. */
  type2tc type = get_uint_type(config.ansi_c.address_width);
  type2tc stype = get_int_type(config.ansi_c.address_width);
  expr2tc op = templ_expr;
  relation_data &rel = static_cast<relation_data &>(*op);
  rel.side_1 = typecast2tc(type, pointer_offset2tc(stype, side1));
  rel.side_2 = typecast2tc(type, pointer_offset2tc(stype, side2));
  return convert_ast(op);
}

smt_astt
smt_convt::convert_pointer_arith(const expr2tc &expr, const type2tc &type)
{
  const arith_2ops &expr_ref = static_cast<const arith_2ops &>(*expr);
  const expr2tc &side1 = expr_ref.side_1;
  const expr2tc &side2 = expr_ref.side_2;

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
  op1_is_ptr = (is_pointer_type(side1)) ? 2 : 0;
  op2_is_ptr = (is_pointer_type(side2)) ? 1 : 0;

  switch (ret_is_ptr | op1_is_ptr | op2_is_ptr)
  {
  case 0:
    assert(false);
    break;
  case 3:
  case 7:
    // The C spec says that we're allowed to subtract two pointers to get
    // the offset of one from the other. However, they absolutely have to
    // point at the same data object, or it's undefined operation. This is
    // already asserted for elsewhere.
    if (expr->expr_id == expr2t::sub_id)
    {
      expr2tc offs1 = pointer_offset2tc(signed_size_type2(), side1);
      expr2tc offs2 = pointer_offset2tc(signed_size_type2(), side2);
      expr2tc the_ptr_offs = sub2tc(offs1->type, offs1, offs2);

      if (ret_is_ptr)
      {
        // Update field in tuple.
        smt_astt the_ptr = convert_ast(side1);
        return the_ptr->update(this, convert_ast(the_ptr_offs), 1);
      }

      assert(side1->type == side2->type);
      expr2tc type_size =
        type_byte_size_expr(to_pointer_type(side1->type).subtype, &ns);
      type_size = typecast2tc(the_ptr_offs->type, type_size); // diff is signed
      expr2tc ptr_diff = div2tc(the_ptr_offs->type, the_ptr_offs, type_size);

      return convert_ast(ptr_diff);
    }
    else
    {
      log_error("Pointer arithmetic with two pointer operands");
      abort();
    }
    break;
  case 4:
    // Artithmetic operation that has the result type of ptr.
    // Should have been handled at a higher level
    log_error("Non-pointer op being interpreted as pointer without cast");
    abort();
    break;
  case 1:
  case 2:
  { // Block required to give a variable lifetime to the cast/add variables
    expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
    expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

    expr2tc add = add2tc(ptr_op->type, ptr_op, non_ptr_op);
    // That'll generate the correct pointer arithmetic; now typecast
    expr2tc cast = typecast2tc(type, add);
    return convert_ast(cast);
  }
  case 5:
  case 6:
  {
    expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
    expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

    // Actually perform some pointer arith
    const pointer_type2t &ptr_type = to_pointer_type(ptr_op->type);
    expr2tc pointee_size = type_byte_size_expr(ptr_type.subtype, &ns);
    type2tc inttype = machine_ptr;
    type2tc difftype = get_int_type(config.ansi_c.address_width);

    if (non_ptr_op->type->get_width() != config.ansi_c.pointer_width())
      non_ptr_op = typecast2tc(machine_ptr, non_ptr_op);

    expr2tc mul = mul2tc(inttype, non_ptr_op, pointee_size);

    // Add or sub that value
    expr2tc ptr_offset =
      typecast2tc(inttype, pointer_offset2tc(difftype, ptr_op));

    expr2tc newexpr;
    if (is_add2t(expr))
    {
      newexpr = add2tc(inttype, mul, ptr_offset);
    }
    else
    {
      // Preserve order for subtraction.
      expr2tc tmp_op1 = (op1_is_ptr) ? ptr_offset : mul;
      expr2tc tmp_op2 = (op1_is_ptr) ? mul : ptr_offset;
      newexpr = sub2tc(inttype, tmp_op1, tmp_op2);
    }

    // Voila, we have our pointer arithmatic
    smt_astt the_ptr = convert_ast(ptr_op);

    simplify(newexpr);

    // That calculated the offset; update field in pointer.
    return the_ptr->update(this, convert_ast(newexpr), 1);
  }
  }

  log_error("Fell through convert_pointer_logic");
  abort();
}

void smt_convt::renumber_symbol_address(
  const expr2tc &guard,
  const expr2tc &addr_symbol,
  const expr2tc &new_size)
{
  const symbol2t &sym = to_symbol2t(addr_symbol);
  std::string str = sym.get_symbol_name();

  const typet *t = nullptr;
  if (const symbolt *s = ns.lookup(sym.thename))
    t = &s->type;

  // Two different approaches if we do or don't have an address-of pointer
  // variable already.

  renumber_mapt::iterator it = renumber_map.back().find(str);
  if (it != renumber_map.back().end())
  {
    // There's already an address-of variable for this pointer. Set up a new
    // object number, and nondeterministically pick the new value.

    unsigned int new_obj_num = pointer_logic.back().get_free_obj_num();
    smt_astt output = init_pointer_obj(new_obj_num, new_size, t);

    // Now merge with the old value for all future address-of's

    it->second = output->ite(this, convert_ast(guard), it->second);
  }
  else
  {
    // Newly bumped pointer. Still needs a new number though.
    unsigned int obj_num = pointer_logic.back().get_free_obj_num();
    smt_astt output = init_pointer_obj(obj_num, new_size, t);

    // Store in renumbered store.
    renumber_mapt::value_type v(str, output);
    renumber_map.back().insert(v);
  }
}

smt_astt smt_convt::convert_identifier_pointer(
  const expr2tc &expr,
  const std::string &symbol,
  const typet *type)
{
  smt_astt a;
  std::string cte, identifier;
  unsigned int obj_num;

  if (!ptr_foo_inited)
  {
    log_error(
      "SMT solver must call smt_post_init immediately after construction");
    abort();
  }

  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    if (sym.thename == "NULL")
    {
      // For null, other pieces of code will have already initialized its
      // value, so we can just refer to a symbol.
      type2tc t = pointer_type2tc(get_empty_type());
      a = tuple_api->mk_tuple_symbol(symbol, convert_sort(t));

      return a;
    }
  }

  // Construct canonical address-of this thing, and check the cache. The addrof
  // expression this is sourced from might have ended up with the wrong type,
  // alas.
  expr2tc new_addr_of = address_of2tc(expr->type, expr);
  smt_cachet::const_iterator cache_result = smt_cache.find(new_addr_of);
  if (cache_result != smt_cache.end())
    return cache_result->ast;

  // Has this been touched by realloc / been re-numbered?
  renumber_mapt::iterator it = renumber_map.back().find(symbol);
  if (it != renumber_map.back().end())
  {
    // Yes -- take current obj num and we're done.
    return it->second;
  }

  // New object. add_object won't duplicate objs for identical exprs
  // (it's a map)
  obj_num = pointer_logic.back().add_object(expr);

  // Produce a symbol representing this.
  type2tc t = pointer_type2tc(get_empty_type());
  a = tuple_api->mk_tuple_symbol(symbol, convert_sort(t));

  // If this object hasn't yet been put in the address space record, we need to
  // assert that the symbol has the object ID we've allocated, and then fill out
  // the address space record.
  if (addr_space_data.back().find(obj_num) == addr_space_data.back().end())
  {
    // Fetch a size.
    type2tc ptr_loc_type = size_type2();
    expr2tc size;
    try
    {
      size = type_byte_size_expr(expr->type, &ns);
    }
    catch (const array_type2t::inf_sized_array_excp &e)
    {
      // This can occur when external symbols with no known size are used.
      // in that case, make a reasonable assumption on how large they might be,
      // say, 64k.
      size = constant_int2tc(ptr_loc_type, BigInt(0x10000));
    }

    smt_astt output = init_pointer_obj(obj_num, size, type);
    assert_ast(a->eq(this, output));
  }

  // Insert canonical address-of this expression.
  struct smt_cache_entryt entry = {new_addr_of, a, ctx_level};
  smt_cache.insert(entry);

  return a;
}

smt_astt smt_convt::init_pointer_obj(
  unsigned int obj_num,
  const expr2tc &size,
  const typet *type)
{
  std::vector<expr2tc> membs;
  const struct_type2t &ptr_struct = to_struct_type(pointer_struct);
  membs.push_back(constant_int2tc(ptr_struct.members[0], BigInt(obj_num)));
  membs.push_back(constant_int2tc(ptr_struct.members[1], BigInt(0)));
  if (config.ansi_c.cheri)
    membs.push_back(
      constant_int2tc(ptr_struct.members[2], BigInt(0))); /* CHERI-TODO */
  expr2tc ptr_val_s = constant_struct2tc(pointer_struct, membs);
  smt_astt ptr_val = tuple_api->tuple_create(ptr_val_s);

  type2tc ptr_loc_type = ptraddr_type2();

  std::stringstream sse1, sse2;
  sse1 << "__ESBMC_ptr_obj_start_" << obj_num;
  sse2 << "__ESBMC_ptr_obj_end_" << obj_num;
  std::string start_name = sse1.str();
  std::string end_name = sse2.str();

  expr2tc start_sym = symbol2tc(ptr_loc_type, start_name);
  expr2tc end_sym = symbol2tc(ptr_loc_type, end_name);

  /* The accessible object spans addresses [start, end), including start,
   * excluding end. The addresses reserved for this object however are
   * [start, end] including 'end'. The reason is that the "one-past end" pointer
   * still needs to be assigned to this object. Including one more byte at the
   * end has several benefits:
   * - The "one-past end" pointer is assigned to this object, as it should
   *   according to C, see the comment about pointer equality in convert_ast().
   * - It avoids the situation that two different objects are laid out
   *   contiguously in the address space and that an address could legally
   *   match both objects, which the if-then-else chain in
   *   convert_typecast_to_ptr() doesn't support.
   * - Zero-size objects still have one valid address value, so typecasts work.
   */
  expr2tc the_size = typecast2tc(ptr_loc_type, size);
  expr2tc end_value = add2tc(ptr_loc_type, start_sym, the_size);
  expr2tc endisequal = equality2tc(end_value, end_sym);

  // Assert that start + size == end
  assert_expr(endisequal);

  // Even better, if we're operating in bitvector mode, it's possible that
  // the solver will try to be clever and arrange the pointer range to cross
  // the end of the address space (ie, wrap around). So, also assert that
  // end >= start
  expr2tc no_wraparound = greaterthanequal2tc(end_sym, start_sym);
  assert_expr(no_wraparound);

  if (type)
  {
    const irept &alignment = type->find("alignment");
    if (alignment.is_not_nil())
    {
      expr2tc alignment2;
      migrate_expr(static_cast<const exprt &>(alignment), alignment2);
      assert(is_constant_int2t(alignment2));
      alignment2 = typecast2tc(ptr_loc_type, alignment2);
      expr2tc zero = gen_zero(ptr_loc_type);
      expr2tc mod = modulus2tc(ptr_loc_type, start_sym, alignment2);
      expr2tc mod_is_zero = equality2tc(mod, zero);
      assert_expr(mod_is_zero);
    }
  }

  // Generate address space layout constraints.
  finalize_pointer_chain(obj_num);

  addr_space_data.back()[obj_num] = 0; // XXX -- nothing uses this data?

  membs.clear();
  membs.push_back(start_sym);
  membs.push_back(end_sym);
  expr2tc range_struct = constant_struct2tc(addr_space_type, membs);
  std::stringstream ss;
  ss << "__ESBMC_ptr_addr_range_" << obj_num;
  expr2tc range_sym = symbol2tc(addr_space_type, ss.str());
  expr2tc eq = equality2tc(range_sym, range_struct);
  assert_expr(eq);

  // Update array
  bump_addrspace_array(obj_num, range_struct);

  return ptr_val;
}

void smt_convt::finalize_pointer_chain(unsigned int objnum)
{
  type2tc inttype = ptraddr_type2();
  unsigned int num_ptrs = addr_space_data.back().size();
  if (num_ptrs == 0)
    return;

  std::stringstream start1, end1;
  start1 << "__ESBMC_ptr_obj_start_" << objnum;
  end1 << "__ESBMC_ptr_obj_end_" << objnum;
  expr2tc start_i = symbol2tc(inttype, start1.str());
  expr2tc end_i = symbol2tc(inttype, end1.str());

  for (unsigned int j = 0; j < objnum; j++)
  {
    // Obj1 is designed to overlap
    if (j == 1)
      continue;

    std::stringstream startj, endj;
    startj << "__ESBMC_ptr_obj_start_" << j;
    endj << "__ESBMC_ptr_obj_end_" << j;
    expr2tc start_j = symbol2tc(inttype, startj.str());
    expr2tc end_j = symbol2tc(inttype, endj.str());

    // Formula: (i_end < j_start) || (i_start > j_end)
    // Previous assertions ensure start <= end for all objs.
    expr2tc lt1 = lessthan2tc(end_i, start_j);
    expr2tc gt1 = greaterthan2tc(start_i, end_j);
    expr2tc no_overlap = or2tc(lt1, gt1);

    expr2tc e = no_overlap;

    /* If a `__ESBMC_alloc` has already been seen, we use it to make the address
     * space constraints on all objects except NULL (j == 0) and INVALID
     * (j == 1) dependent on whether the object is still alive:
     *   (__ESBMC_alloc[j] == true) => (i_end < j_start || i_start > j_end)
     * In case the object j was free'd, it no longer restricts the addresses of
     * the new object i.
     *
     * XXXfbrausse: This is crucially relies on the fact that the current
     * version of the __ESBMC_alloc symbol stored in `current_valid_objects_sym`
     * is the one this new object i gets registered with.
     */
    if (j && current_valid_objects_sym)
    {
      expr2tc alive =
        index2tc(get_bool_type(), current_valid_objects_sym, gen_ulong(j));

      // Tong: "alive" is only changed when the pointer is dynamic from malloc/free.
      // And it's value is always false for some pointers that represent race-flags.
      // However usually this issue will not be exposed because the slicer has sliced
      // away "alive", but it's exposed in no-slice and incremental-smt.
      // For now we just modify for races check.

      if (options.get_bool_option("data-races-check") && cur_dynamic)
      {
        expr2tc dynamic = index2tc(get_bool_type(), cur_dynamic, gen_ulong(j));
        e = implies2tc((or2tc(not2tc(dynamic), alive)), e);
      }
      else
        e = implies2tc(alive, e);
    }

    assert_expr(e);
  }
}

smt_astt smt_convt::convert_addr_of(const expr2tc &expr)
{
  const address_of2t &obj = to_address_of2t(expr);

  std::string symbol_name, out;

  if (is_index2t(obj.ptr_obj) || is_member2t(obj.ptr_obj))
  {
    // This might be a composite index/member/blah chain
    expr2tc offs = compute_pointer_offset(obj.ptr_obj);
    expr2tc base = get_base_object(obj.ptr_obj);

    expr2tc addrof = address_of2tc(obj.type, base);
    smt_astt a = convert_ast(addrof);

    /* constant 1 refers to member 'pointer_offset' of 'pointer_struct' */
    // Update pointer offset to offset to that field.
    return a->update(this, convert_ast(offs), 1);
  }

  if (is_symbol2t(obj.ptr_obj))
  {
    const symbol2t &symbol = to_symbol2t(obj.ptr_obj);

    const typet *t = nullptr;
    if (const symbolt *s = ns.lookup(symbol.thename))
      t = &s->type;

    return convert_identifier_pointer(obj.ptr_obj, symbol.get_symbol_name(), t);
  }

  if (is_constant_string2t(obj.ptr_obj))
  {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    const constant_string2t &str = to_constant_string2t(obj.ptr_obj);
    std::string identifier =
      "address_of_str_const(" + str.value.as_string() + ")";

    // XXX Oh look -- this is vulnerable to the poison null byte.
    std::replace(identifier.begin(), identifier.end(), '\0', '_');

    return convert_identifier_pointer(obj.ptr_obj, identifier, nullptr);
  }

  if (is_constant_array2t(obj.ptr_obj))
  {
    // This can occur (rather than being a constant string) when the C++
    // frontend performs const propagation in functions that pass around
    // character array references/pointers, but it drops some type information
    // along the way.
    // The pointer will remain consistent because any pointer taken to the
    // same constant array will be picked up in the expression cache
    static unsigned int constarr_num = 0;
    std::stringstream ss;
    ss << "address_of_arr_const(" << constarr_num++ << ")";
    return convert_identifier_pointer(obj.ptr_obj, ss.str(), nullptr);
  }

  if (is_if2t(obj.ptr_obj))
  {
    // We can't nondeterministically take the address of something; So instead
    // rewrite this to be if (cond) ? &a : &b;.

    const if2t &ifval = to_if2t(obj.ptr_obj);

    expr2tc addrof1 = address_of2tc(obj.type, ifval.true_value);
    expr2tc addrof2 = address_of2tc(obj.type, ifval.false_value);
    expr2tc newif = if2tc(obj.type, ifval.cond, addrof1, addrof2);
    return convert_ast(newif);
  }

  if (is_typecast2t(obj.ptr_obj))
  {
    // Take the address of whatevers being casted. Either way, they all end up
    // being of a pointer_tuple type, so this should be fine.
    expr2tc tmp = address_of2tc(type2tc(), to_typecast2t(obj.ptr_obj).from);
    tmp->type = obj.type;
    return convert_ast(tmp);
  }

  log_error("Unrecognized address_of operand:\n{}", *expr);
  abort();
}

void smt_convt::init_addr_space_array()
{
  addr_space_sym_num.back() = 1;

  type2tc ptr_int_type = ptraddr_type2(); /* CHERI-TODO */
  expr2tc zero_ptr_int = constant_int2tc(ptr_int_type, BigInt(0));
  expr2tc one_ptr_int = constant_int2tc(ptr_int_type, BigInt(1));
  expr2tc obj1_end_const =
    constant_int2tc(ptr_int_type, BigInt::power2m1(ptr_int_type->get_width()));

  expr2tc obj0_start = symbol2tc(ptr_int_type, "__ESBMC_ptr_obj_start_0");
  expr2tc obj0_end = symbol2tc(ptr_int_type, "__ESBMC_ptr_obj_end_0");

  assert_expr(equality2tc(obj0_start, zero_ptr_int));
  assert_expr(equality2tc(obj0_end, zero_ptr_int));

  expr2tc obj1_start = symbol2tc(ptr_int_type, "__ESBMC_ptr_obj_start_1");
  expr2tc obj1_end = symbol2tc(ptr_int_type, "__ESBMC_ptr_obj_end_1");

  assert_expr(equality2tc(obj1_start, one_ptr_int));
  assert_expr(equality2tc(obj1_end, obj1_end_const));

  expr2tc addr0_tuple = constant_struct2tc(
    addr_space_type, std::vector<expr2tc>{obj0_start, obj0_end});
  assert_expr(equality2tc(
    symbol2tc(addr_space_type, "__ESBMC_ptr_addr_range_0"), addr0_tuple));

  expr2tc addr1_tuple = constant_struct2tc(
    addr_space_type, std::vector<expr2tc>{obj1_start, obj1_end});
  assert_expr(equality2tc(
    symbol2tc(addr_space_type, "__ESBMC_ptr_addr_range_1"), addr1_tuple));

  bump_addrspace_array(pointer_logic.back().get_null_object(), addr0_tuple);
  bump_addrspace_array(pointer_logic.back().get_invalid_object(), addr1_tuple);

  const struct_type2t &ptr_struct = to_struct_type(pointer_struct);

  std::vector<expr2tc> null_members =
                         {
                           constant_int2tc(ptr_struct.members[0], 0),
                           constant_int2tc(ptr_struct.members[1], 0),
                         },
                       inv_members = {
                         constant_int2tc(ptr_struct.members[0], 1),
                         constant_int2tc(ptr_struct.members[1], 0),
                       };
  if (config.ansi_c.cheri)
  {
    null_members.emplace_back(constant_int2tc(ptr_struct.members[2], 0));
    /* same as NULL capability */
    inv_members.emplace_back(constant_int2tc(ptr_struct.members[2], 0));
  }
  expr2tc null_ptr_tuple = constant_struct2tc(pointer_struct, null_members);
  expr2tc invalid_ptr_tuple = constant_struct2tc(pointer_struct, inv_members);

  null_ptr_ast = convert_ast(null_ptr_tuple);
  invalid_ptr_ast = convert_ast(invalid_ptr_tuple);

  // Give value to 'NULL', 'INVALID' symbols
  assert_expr(equality2tc(symbol2tc(pointer_struct, "NULL"), null_ptr_tuple));
  assert_expr(
    equality2tc(symbol2tc(pointer_struct, "INVALID"), invalid_ptr_tuple));

  addr_space_data.back()[0] = 0;
  addr_space_data.back()[1] = 0;
}

void smt_convt::bump_addrspace_array(unsigned int idx, const expr2tc &val)
{
  expr2tc oldname = symbol2tc(
    addr_space_arr_type,
    "__ESBMC_addrspace_arr_" + std::to_string(addr_space_sym_num.back()++));
  expr2tc store = with2tc(
    addr_space_arr_type,
    oldname,
    constant_int2tc(machine_ptr, BigInt(idx)),
    val);
  expr2tc newname = symbol2tc(
    addr_space_arr_type,
    "__ESBMC_addrspace_arr_" + std::to_string(addr_space_sym_num.back()));
  convert_assign(equality2tc(newname, store));
}

std::string smt_convt::get_cur_addrspace_ident()
{
  std::stringstream ss;
  ss << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back();
  return ss.str();
}
