#include <algorithm>
#include <sstream>

#include "smt_conv.h"

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

smt_astt
smt_convt::convert_ptr_cmp(const expr2tc &side1, const expr2tc &side2,
                           const expr2tc &templ_expr)
{
  // Special handling for pointer comparisons (both ops are pointers; otherwise
  // it's obviously broken). First perform a test as to whether or not the
  // pointer locations are greater or lower; and only involve the ptr offset
  // if the ptr objs are the same.
  type2tc int_type = machine_ptr;

  pointer_object2tc ptr_obj1(int_type, side1);
  pointer_offset2tc ptr_offs1(int_type, side1);
  pointer_object2tc ptr_obj2(int_type, side2);
  pointer_offset2tc ptr_offs2(int_type, side2);

  symbol2tc addrspacesym(addr_space_arr_type, get_cur_addrspace_ident());
  index2tc obj1_data(addr_space_type, addrspacesym, ptr_obj1);
  index2tc obj2_data(addr_space_type, addrspacesym, ptr_obj2);

  member2tc obj1_start(int_type, obj1_data, irep_idt("start"));
  member2tc obj2_start(int_type, obj2_data, irep_idt("start"));

  expr2tc start_expr = templ_expr, offs_expr = templ_expr;

  // To ensure we can do this in an operation independant way, we're going to
  // clone the original comparison expression, and replace its operands with
  // new values. Works whatever the expr is, so long as it has two operands.
  *start_expr.get()->get_sub_expr_nc(0) = obj1_start;
  *start_expr.get()->get_sub_expr_nc(1) = obj2_start;
  *offs_expr.get()->get_sub_expr_nc(0) = ptr_offs1;
  *offs_expr.get()->get_sub_expr_nc(1) = ptr_offs2;

  // Those are now boolean type'd relations.
  equality2tc is_same_obj_expr(ptr_obj1, ptr_obj2);

  if2tc res(offs_expr->type, is_same_obj_expr, offs_expr, start_expr);
  return convert_ast(res);
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

  switch (ret_is_ptr | op1_is_ptr | op2_is_ptr) {
    case 0:
      assert(false);
      break;
    case 3:
    case 7:
      // The C spec says that we're allowed to subtract two pointers to get
      // the offset of one from the other. However, they absolutely have to
      // point at the same data object, or it's undefined operation. This is
      // already asserted for elsewhere.
      if (expr->expr_id == expr2t::sub_id) {
        pointer_offset2tc offs1(machine_ptr, side1);
        pointer_offset2tc offs2(machine_ptr, side2);
        sub2tc the_ptr_offs(offs1->type, offs1, offs2);

        if (ret_is_ptr) {
          // Update field in tuple.
          smt_astt the_ptr = convert_ast(side1);
          return the_ptr->update(this, convert_ast(the_ptr_offs), 1);
        } else {
          return convert_ast(the_ptr_offs);
        }
      } else {
        std::cerr << "Pointer arithmetic with two pointer operands"
                  << std::endl;
        abort();
      }
      break;
    case 4:
      // Artithmetic operation that has the result type of ptr.
      // Should have been handled at a higher level
      std::cerr << "Non-pointer op being interpreted as pointer without cast"
                << std::endl;
      abort();
      break;
    case 1:
    case 2:
      { // Block required to give a variable lifetime to the cast/add variables
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      add2tc add(ptr_op->type, ptr_op, non_ptr_op);
      // That'll generate the correct pointer arithmatic; now typecast
      typecast2tc cast(type, add);
      return convert_ast(cast);
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
      mp_integer type_size;
      if (is_empty_type(followed_type))
        type_size = 1;
      else
        type_size = type_byte_size(followed_type);

      // Generate nonptr * constant.
      type2tc inttype = machine_ptr;
      constant_int2tc constant(inttype, type_size);

      if (non_ptr_op->type->get_width() < config.ansi_c.pointer_width)
        non_ptr_op = typecast2tc(machine_ptr, non_ptr_op);

      expr2tc mul = mul2tc(inttype, non_ptr_op, constant);

      // Add or sub that value
      expr2tc ptr_offset = pointer_offset2tc(inttype, ptr_op);

      expr2tc newexpr;
      if (is_add2t(expr)) {
        newexpr = add2tc(inttype, mul, ptr_offset);
      } else {
        // Preserve order for subtraction.
        expr2tc tmp_op1 = (op1_is_ptr) ? ptr_offset : mul;
        expr2tc tmp_op2 = (op1_is_ptr) ? mul : ptr_offset;
        newexpr = sub2tc(inttype, tmp_op1, tmp_op2);
      }

      // Voila, we have our pointer arithmatic
      smt_astt the_ptr = convert_ast(ptr_op);

      expr2tc tmp = newexpr->simplify();
      if (!is_nil_expr(tmp))
        newexpr = tmp;

      // That calculated the offset; update field in pointer.
      return the_ptr->update(this, convert_ast(newexpr), 1);
    }
  }

  std::cerr << "Fell through convert_pointer_logic" << std::endl;
  abort();
}

void
smt_convt::renumber_symbol_address(
  const expr2tc &guard,
  const expr2tc &addr_symbol,
  const expr2tc &new_size)
{
  const symbol2t &sym = to_symbol2t(addr_symbol);
  std::string str = sym.get_symbol_name();

  // Two different approaches if we do or don't have an address-of pointer
  // variable already.

  renumber_mapt::iterator it = renumber_map.back().find(str);
  if (it != renumber_map.back().end()) {
    // There's already an address-of variable for this pointer. Set up a new
    // object number, and nondeterministically pick the new value.

    unsigned int new_obj_num = pointer_logic.back().get_free_obj_num();
    smt_astt output = init_pointer_obj(new_obj_num, new_size);

    // Now merge with the old value for all future address-of's

    it->second = mk_func_app(output->sort, SMT_FUNC_ITE,
                             convert_ast(guard), output, it->second);
  } else {
    // Newly bumped pointer. Still needs a new number though.
    unsigned int obj_num = pointer_logic.back().get_free_obj_num();
    smt_astt output = init_pointer_obj(obj_num, new_size);

    // Store in renumbered store.
    renumber_mapt::value_type v(str, output);
    renumber_map.back().insert(v);
  }
}

smt_astt
smt_convt::convert_identifier_pointer(const expr2tc &expr, std::string symbol)
{
  smt_astt a;
  std::string cte, identifier;
  unsigned int obj_num;

  if (!ptr_foo_inited) {
    std::cerr << "SMT solver must call smt_post_init immediately after "
              << "construction" << std::endl;
    abort();
  }

  if (is_symbol2t(expr)) {
    const symbol2t &sym = to_symbol2t(expr);
    if (sym.thename == "NULL" || sym.thename == "0") {
      // For null, other pieces of code will have already initialized its
      // value, so we can just refer to a symbol.
      obj_num = pointer_logic.back().get_null_object();

      type2tc t(new pointer_type2t(get_empty_type()));
      a = tuple_api->mk_tuple_symbol(symbol, convert_sort(t));

      return a;
    }
  }

  // Construct canonical address-of this thing, and check the cache. The addrof
  // expression this is sourced from might have ended up with the wrong type,
  // alas.
  address_of2tc new_addr_of(expr->type, expr);
  smt_cachet::const_iterator cache_result = smt_cache.find(new_addr_of);
  if (cache_result != smt_cache.end())
    return (cache_result->ast);

  // Has this been touched by realloc / been re-numbered?
  renumber_mapt::iterator it = renumber_map.back().find(symbol);
  if (it != renumber_map.back().end()) {
    // Yes -- take current obj num and we're done.
    return it->second;
  }

  // New object. add_object won't duplicate objs for identical exprs
  // (it's a map)
  obj_num = pointer_logic.back().add_object(expr);

  // Produce a symbol representing this.
  type2tc t(new pointer_type2t(get_empty_type()));
  a = tuple_api->mk_tuple_symbol(symbol, convert_sort(t));

  // If this object hasn't yet been put in the address space record, we need to
  // assert that the symbol has the object ID we've allocated, and then fill out
  // the address space record.
  if (addr_space_data.back().find(obj_num) == addr_space_data.back().end()) {
    // Fetch a size.
    type2tc ptr_loc_type(new unsignedbv_type2t(config.ansi_c.word_size));
    expr2tc size;
    try {
      uint64_t type_size = expr->type->get_width() / 8;
      size = constant_int2tc(ptr_loc_type, BigInt(type_size));
    } catch (array_type2t::dyn_sized_array_excp *e) {
      size = e->size;
    } catch (array_type2t::inf_sized_array_excp *e) {
      // This can occur when external symbols with no known size are used.
      // in that case, make a reasonable assumption on how large they might be,
      // say, 64k.
      size = constant_int2tc(ptr_loc_type, BigInt(0x10000));
    } catch (type2t::symbolic_type_excp *e) {
      // Type is empty or code -- something that we can never have a real size
      // for. In that case, create an object of size 1: this means we have a
      // valid entry in the address map, but that any modification of the
      // pointer leads to invalidness, because there's no size to think about.
      size = constant_int2tc(ptr_loc_type, BigInt(1));
    }

    smt_astt output = init_pointer_obj(obj_num, size);
    smt_astt args[2];
    args[0] = a;
    args[1] = output;

    assert_ast(args[0]->eq(this, args[1]));
  }

  // Insert canonical address-of this expression.
  struct smt_cache_entryt entry = { new_addr_of, a, ctx_level };
  smt_cache.insert(entry);

  return a;
}

smt_astt
smt_convt::init_pointer_obj(unsigned int obj_num, const expr2tc &size)
{
    std::vector<expr2tc> membs;
    membs.push_back(constant_int2tc(machine_ptr, BigInt(obj_num)));
    membs.push_back(constant_int2tc(machine_ptr, BigInt(0)));
    constant_struct2tc ptr_val_s(pointer_struct, membs);
    smt_astt ptr_val = tuple_api->tuple_create(ptr_val_s);

    type2tc ptr_loc_type = machine_ptr;

    std::stringstream sse1, sse2;
    sse1 << "__ESBMC_ptr_obj_start_" << obj_num;
    sse2 << "__ESBMC_ptr_obj_end_" << obj_num;
    std::string start_name = sse1.str();
    std::string end_name = sse2.str();

    symbol2tc start_sym(ptr_loc_type, start_name);
    symbol2tc end_sym(ptr_loc_type, end_name);

    // Another thing to note is that the end var must be /the size of the obj/
    // from start. Express this in irep.
    expr2tc endisequal;
    expr2tc the_offs;
    the_offs = typecast2tc(machine_ptr, size);
    add2tc start_plus_offs(ptr_loc_type, start_sym, the_offs);
    endisequal = equality2tc(start_plus_offs, end_sym);

    // Assert that start + offs == end
    assert_expr(endisequal);

    // Even better, if we're operating in bitvector mode, it's possible that
    // the solver will try to be clever and arrange the pointer range to cross
    // the end of the address space (ie, wrap around). So, also assert that
    // end > start
    // Except when the size is zero, which might not be statically dicoverable
    constant_int2tc zero_val(the_offs->type, BigInt(0));
    equality2tc zeroeq(zero_val, the_offs);
    greaterthan2tc wraparound(end_sym, start_sym);
    assert_expr(or2tc(zeroeq, wraparound));

    // Generate address space layout constraints.
    finalize_pointer_chain(obj_num);

    addr_space_data.back()[obj_num] = 0; // XXX -- nothing uses this data?

    membs.clear();
    membs.push_back(start_sym);
    membs.push_back(end_sym);
    constant_struct2tc range_struct(addr_space_type, membs);
    std::stringstream ss;
    ss << "__ESBMC_ptr_addr_range_" <<  obj_num;
    symbol2tc range_sym(addr_space_type, ss.str());
    equality2tc eq(range_sym, range_struct);
    assert_expr(eq);

    // Update array
    bump_addrspace_array(obj_num, range_struct);

    // Finally, ensure that the array storing whether this pointer is dynamic,
    // is initialized for this ptr to false. That way, only pointers created
    // through malloc will be marked dynamic.

    type2tc arrtype(new array_type2t(type2tc(new bool_type2t()),
                                     expr2tc((expr2t*)NULL), true));
    symbol2tc allocarr(arrtype, dyn_info_arr_name);
    constant_int2tc objid(machine_uint, BigInt(obj_num));
    index2tc idx(get_bool_type(), allocarr, objid);
    equality2tc dyn_eq(idx, false_expr);
    assert_expr(dyn_eq);

    return ptr_val;
}

void
smt_convt::finalize_pointer_chain(unsigned int objnum)
{
  type2tc inttype = machine_ptr;
  unsigned int num_ptrs = addr_space_data.back().size();
  if (num_ptrs == 0)
    return;

  std::stringstream start1, end1;
  start1 << "__ESBMC_ptr_obj_start_" << objnum;
  end1 << "__ESBMC_ptr_obj_end_" << objnum;
  symbol2tc start_i(inttype, start1.str());
  symbol2tc end_i(inttype, end1.str());

  for (unsigned int j = 0; j < objnum; j++) {
    // Obj1 is designed to overlap
    if (j == 1)
      continue;

    std::stringstream startj, endj;
    startj << "__ESBMC_ptr_obj_start_" << j;
    endj << "__ESBMC_ptr_obj_end_" << j;
    symbol2tc start_j(inttype, startj.str());
    symbol2tc end_j(inttype, endj.str());

    // Formula: (i_end < j_start) || (i_start > j_end)
    // Previous assertions ensure start < end for all objs.
    lessthan2tc lt1(end_i, start_j);
    greaterthan2tc gt1(start_i, end_j);
    or2tc or1(lt1, gt1);
    assert_expr(or1);
  }

  return;
}

smt_astt
smt_convt::convert_addr_of(const expr2tc &expr)
{
  const address_of2t &obj = to_address_of2t(expr);

  std::string symbol_name, out;

  if (is_index2t(obj.ptr_obj)) {
    // This might be a composite index/member/blah chain
    expr2tc offs = compute_pointer_offset(obj.ptr_obj);
    expr2tc base = get_base_object(obj.ptr_obj);

    address_of2tc addrof(obj.type, base);
    smt_astt a = convert_ast(addrof);
    return a->update(this, convert_ast(offs), 1);
  } else if (is_member2t(obj.ptr_obj)) {
    expr2tc offs = compute_pointer_offset(obj.ptr_obj);
    expr2tc base = get_base_object(obj.ptr_obj);

    address_of2tc addr(obj.type, base);

    smt_astt a = convert_ast(addr);

    // Update pointer offset to offset to that field.
    return a->update(this, convert_ast(offs), 1);
  } else if (is_symbol2t(obj.ptr_obj)) {
    const symbol2t &symbol = to_symbol2t(obj.ptr_obj);
    return convert_identifier_pointer(obj.ptr_obj, symbol.get_symbol_name());
  } else if (is_constant_string2t(obj.ptr_obj)) {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    const constant_string2t &str = to_constant_string2t(obj.ptr_obj);
    std::string identifier =
      "address_of_str_const(" + str.value.as_string() + ")";

    // XXX Oh look -- this is vulnerable to the poison null byte.
    std::replace(identifier.begin(), identifier.end(), '\0', '_');

    return convert_identifier_pointer(obj.ptr_obj, identifier);
  } else if (is_constant_array2t(obj.ptr_obj)) {
    // This can occur (rather than being a constant string) when the C++
    // frontend performs const propagation in functions that pass around
    // character array references/pointers, but it drops some type information
    // along the way.
    // The pointer will remain consistent because any pointer taken to the
    // same constant array will be picked up in the expression cache
    static unsigned int constarr_num = 0;
    std::stringstream ss;
    ss << "address_of_arr_const(" << constarr_num++ << ")";
    return convert_identifier_pointer(obj.ptr_obj, ss.str());
  } else if (is_if2t(obj.ptr_obj)) {
    // We can't nondeterministically take the address of something; So instead
    // rewrite this to be if (cond) ? &a : &b;.

    const if2t &ifval = to_if2t(obj.ptr_obj);

    address_of2tc addrof1(obj.type, ifval.true_value);
    address_of2tc addrof2(obj.type, ifval.false_value);
    if2tc newif(obj.type, ifval.cond, addrof1, addrof2);
    return convert_ast(newif);
  } else if (is_typecast2t(obj.ptr_obj)) {
    // Take the address of whatevers being casted. Either way, they all end up
    // being of a pointer_tuple type, so this should be fine.
    address_of2tc tmp(type2tc(), to_typecast2t(obj.ptr_obj).from);
    tmp.get()->type = obj.type;
    return convert_ast(tmp);
  }

  std::cerr << "Unrecognized address_of operand:" << std::endl;
  expr->dump();
  abort();
}


void
smt_convt::init_addr_space_array(void)
{
  addr_space_sym_num.back() = 1;

  type2tc ptr_int_type = machine_ptr;
  constant_int2tc zero_ptr_int(ptr_int_type, BigInt(0));
  constant_int2tc one_ptr_int(ptr_int_type, BigInt(1));
  BigInt allones((config.ansi_c.pointer_width == 32)
                  ? 0xFFFFFFFF
                  : 0xFFFFFFFFFFFFFFFFULL);
  constant_int2tc obj1_end_const(ptr_int_type, allones);

  symbol2tc obj0_start(ptr_int_type, "__ESBMC_ptr_obj_start_0");
  symbol2tc obj0_end(ptr_int_type, "__ESBMC_ptr_obj_end_0");
  equality2tc obj0_start_eq(obj0_start, zero_ptr_int);
  equality2tc obj0_end_eq(obj0_end, zero_ptr_int);

  assert_expr(obj0_start_eq);
  assert_expr(obj0_end_eq);

  symbol2tc obj1_start(ptr_int_type, "__ESBMC_ptr_obj_start_1");
  symbol2tc obj1_end(ptr_int_type, "__ESBMC_ptr_obj_end_1");
  equality2tc obj1_start_eq(obj1_start, one_ptr_int);
  equality2tc obj1_end_eq(obj1_end, obj1_end_const);

  assert_expr(obj1_start_eq);
  assert_expr(obj1_end_eq);

  std::vector<expr2tc> membs;
  membs.push_back(obj0_start);
  membs.push_back(obj0_end);
  constant_struct2tc addr0_tuple(addr_space_type, membs);
  symbol2tc addr0_range(addr_space_type, "__ESBMC_ptr_addr_range_0");
  equality2tc addr0_range_eq(addr0_range, addr0_tuple);
  assert_expr(addr0_range_eq);

  membs.clear();
  membs.push_back(obj1_start);
  membs.push_back(obj1_end);
  constant_struct2tc addr1_tuple(addr_space_type, membs);
  symbol2tc addr1_range(addr_space_type, "__ESBMC_ptr_addr_range_1");
  equality2tc addr1_range_eq(addr1_range, addr1_tuple);
  assert_expr(addr1_range_eq);

  bump_addrspace_array(pointer_logic.back().get_null_object(), addr0_tuple);
  bump_addrspace_array(pointer_logic.back().get_invalid_object(), addr1_tuple);

  // Give value to '0', 'NULL', 'INVALID' symbols
  symbol2tc zero_ptr(pointer_struct, "0");
  symbol2tc null_ptr(pointer_struct, "NULL");
  symbol2tc invalid_ptr(pointer_struct, "INVALID");

  membs.clear();
  membs.push_back(zero_ptr_int);
  membs.push_back(zero_ptr_int);
  constant_struct2tc null_ptr_tuple(pointer_struct, membs);
  membs.clear();
  membs.push_back(one_ptr_int);
  membs.push_back(zero_ptr_int);
  constant_struct2tc invalid_ptr_tuple(pointer_struct, membs);

  equality2tc zero_eq(zero_ptr, null_ptr_tuple);
  equality2tc null_eq(null_ptr, null_ptr_tuple);
  equality2tc invalid_eq(invalid_ptr, invalid_ptr_tuple);

  null_ptr_ast = convert_ast(null_ptr_tuple);
  invalid_ptr_ast = convert_ast(invalid_ptr_tuple);

  assert_expr(zero_eq);
  assert_expr(null_eq);
  assert_expr(invalid_eq);

  addr_space_data.back()[0] = 0;
  addr_space_data.back()[1] = 0;
}

void
smt_convt::bump_addrspace_array(unsigned int idx, const expr2tc &val)
{
  std::stringstream ss, ss2;
  std::string str, new_str;
  type2tc ptr_int_type = machine_ptr;

  ss << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back()++;
  symbol2tc oldname(addr_space_arr_type, ss.str());
  constant_int2tc ptr_idx(ptr_int_type, BigInt(idx));

  with2tc store(addr_space_arr_type, oldname, ptr_idx, val);
  ss2 << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back();
  symbol2tc newname(addr_space_arr_type, ss2.str());
  equality2tc eq(newname, store);
  convert_assign(eq);
  return;
}

std::string
smt_convt::get_cur_addrspace_ident(void)
{
  std::stringstream ss;
  ss << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back();
  return ss.str();
}

