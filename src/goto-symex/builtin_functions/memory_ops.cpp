#include <cassert>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <string>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/std_types.h>
#include <algorithm>

// Computes the equivalent object value when considering a memset operation on it
static inline expr2tc gen_byte_expression_byte_update(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  // Sadly, our simplifier can not typecast from value operations
  // safely. We can however :)
  auto new_src = src;
  auto new_type = type;

  auto found_constant = false;
  auto optimized = src->simplify();
  if (optimized)
  {
    found_constant = is_typecast2t(optimized) &&
                     is_constant_int2t(to_typecast2t(optimized).from);
    if (found_constant)
    {
      new_src = to_typecast2t(optimized).from;
      new_type = get_int64_type();
    }
  }

  expr2tc result = new_src;
  auto value_downcast = typecast2tc(get_uint8_type(), value);

  expr2tc off = constant_int2tc(get_int32_type(), BigInt(offset));
  for (size_t counter = 0; counter < num_of_bytes; counter++)
  {
    expr2tc increment = constant_int2tc(get_int32_type(), BigInt(counter));
    result = byte_update2tc(
      new_type,
      result,
      add2tc(off->type, off, increment),
      value_downcast,
      false);
  }

  if (found_constant)
    result = typecast2tc(type, result);

  simplify(result);

  return result;
}

// Computes the equivalent object value when considering a memset operation on it
static inline expr2tc gen_byte_expression(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  /**
   * The idea of this expression is to compute the object value
   * in the case where every byte `value` was set up until num_of_bytes
   *
   * Note: this function assumes that all memory checks have been done!
   *
   * In summary, there are two main computations here:
   *
   * A. Generate the byte representation, this is mostly through
   *    the `result` expression. The expression is initialized with zero
   *    and then, until the num_of_bytes is reached it will do a full byte
   *    left-shift followed by an bitor operation with the byte value:
   *
   *    For example, for a integer(4 bytes) with memset using 3 bytes and value 0xF1
   *
   *    step 1: 0x00000000 -- left-shift 8 -- 0x00000000 -- bitor -- 0x000000F1
   *    step 2: 0x000000F1 -- left-shift 8 -- 0x0000F100 -- bitor -- 0x0000F1F1
   *    step 3: 0x0000F1F1 -- left-shift 8 -- 0x00F1F100 -- bitor -- 0x00F1F1F1
   *
   *    Since we only want 3 bytes, the initialized object value would be 0x00F1F1F1
   *
   * B. Generate a mask of the bits that were not set, this is done because skipped bits
   *    need to be returned back. The computation of this is simple, we initialize every
   *    bit that was changed by the byte-representation computation with a 1, which is then
   *    negated to be applied with an bitand in the original value:
   *
   *    Back to the example in A, we had the byte-representation of  0x00F1F1F1. If the
   *    original value was 0xA2A2A2A2, then we would have the following mask:
   *
   *    step 1: 0x00000000 -- set-bits -- 0x000000FF
   *    step 2: 0x000000FF -- set-bits -- 0x0000FFFF
   *    step 3: 0x0000FFFF -- set-bits -- 0x00FFFFFF
   *
   *   So, 0x00FFFFFF is the mask for all bits changed. We can negate it to: 0xFF000000
   *
   *   Then, we can apply it to the original source value with bitand
   *
   *   0xA2A2A2A2 AND 0xFF000000 --> 0xA2000000
   *
   * Finally, we get the result from A and B and unify them through a bitor
   *
   *  0xA2000000 OR 0x00F1F1F1 --> 0xA2F1F1F1
   *
   * Note about offsets: To handle them, we apply left shifts to the remaining offset after
   * the computation of the object-value and initial mask representation
   *
   */

  if (is_pointer_type(type))
    return gen_byte_expression_byte_update(
      type, src, value, num_of_bytes, offset);

  expr2tc result = gen_zero(type);
  auto value_downcast = typecast2tc(get_uint8_type(), value);
  auto value_upcast = typecast2tc(
    type,
    value_downcast); // so smt_conv won't complain about the width of the type

  expr2tc mask = gen_zero(type);

  const auto eight = constant_int2tc(type, BigInt(8));
  const auto one = constant_int2tc(type, BigInt(1));
  for (unsigned i = 0; i < num_of_bytes; i++)
  {
    result = shl2tc(type, result, eight);
    result = bitor2tc(type, result, value_upcast);

    for (int m = 0; m < 8; m++)
    {
      mask = shl2tc(type, mask, one);
      mask = bitor2tc(type, mask, one);
    }
  }

  // Do the rest of the offset!
  for (unsigned i = 0; i < offset; i++)
  {
    result = shl2tc(type, result, eight);
    mask = shl2tc(type, mask, eight);
  }

  mask = bitnot2tc(type, mask);
  mask = bitand2tc(type, src, mask);
  result = bitor2tc(type, result, mask);

  simplify(result);
  return result;
}

static inline expr2tc gen_value_by_byte(
  const type2tc &type,
  const expr2tc &src,
  const expr2tc &value,
  const size_t num_of_bytes,
  const size_t offset)
{
  /**
   * @brief Construct a new object, initializing it with the memset equivalent
   *
   * There are a few corner cases here:
   *
   * 1 - Primitives: these are simple: just generate the byte_expression directly
   * 2 - Arrays: these are ok: just keep generating byte_expression for each member
   *        until a limit has arrived. Dynamic memory is dealt here.
   * 3 - Structs/Union: these are the hardest as we have to take the alignment into
   *        account when dealing with it. Hopefully the clang-frontend already give it
   *        to us.
   *
   */

  if (num_of_bytes == 0)
    return src;

  /* TODO: Bitwise operations are valid for floats, but we don't have an
   * implementation, yet. Give up. */
  if (is_floatbv_type(type) || is_fixedbv_type(type))
    return expr2tc();

  if (is_scalar_type(type) && type->get_width() == 8 && offset == 0)
    return typecast2tc(type, value);

  if (is_array_type(type))
  {
    /*
     * Very straighforward, get the total number_of_bytes and keep subtracting until
     * the end
     */

    expr2tc result = gen_zero(type);
    constant_array2t &data = to_constant_array2t(result);

    uint64_t base_size =
      type_byte_size(to_array_type(type).subtype).to_uint64();
    uint64_t bytes_left = num_of_bytes;
    uint64_t offset_left = offset;

    for (unsigned i = 0; i < data.datatype_members.size(); i++)
    {
      BigInt position(i);
      expr2tc local_member = index2tc(
        to_array_type(type).subtype,
        src,
        constant_int2tc(get_uint32_type(), position));
      // Skip offsets
      if (offset_left >= base_size)
      {
        data.datatype_members[i] = local_member;
        offset_left -= base_size;
      }
      else
      {
        uint64_t bytes_to_write =
          bytes_left < base_size ? bytes_left : base_size;
        data.datatype_members[i] = gen_value_by_byte(
          to_array_type(type).subtype,
          local_member,
          value,
          bytes_to_write,
          offset_left);
        if (!data.datatype_members[i])
          return expr2tc();
        bytes_left =
          bytes_left <= base_size ? 0 : bytes_left - (base_size - offset_left);
        offset_left = 0;
      }
    }

    return result;
  }

  if (is_struct_type(type))
  {
    /** Similar to array, however get the size of
     * each component
     */
    expr2tc result = gen_zero(type);
    constant_struct2t &data = to_constant_struct2t(result);
    uint64_t bytes_left = num_of_bytes;
    uint64_t offset_left = offset;

    for (unsigned i = 0; i < data.datatype_members.size(); i++)
    {
      irep_idt name = to_struct_type(type).member_names[i];
      // TODO: We need a better way to detect bitfields
      if (has_prefix(name.as_string(), "bit_field_pad$"))
        return expr2tc();
      expr2tc local_member =
        member2tc(to_struct_type(type).members[i], src, name);

      // Since it is a symbol, lets start from the old value
      if (is_pointer_type(to_struct_type(type).members[i]))
        data.datatype_members[i] = local_member;

      type2tc current_member_type = data.datatype_members[i]->type;

      uint64_t current_member_size =
        type_byte_size(current_member_type).to_uint64();

      // Skip offsets
      if (offset_left >= current_member_size)
      {
        data.datatype_members[i] = local_member;
        offset_left -= current_member_size;
      }
      else
      {
        assert(offset_left < current_member_size);
        uint64_t bytes_to_write = std::min(bytes_left, current_member_size);
        data.datatype_members[i] = gen_value_by_byte(
          current_member_type,
          local_member,
          value,
          bytes_to_write,
          offset_left);

        if (!data.datatype_members[i])
          return expr2tc();

        bytes_left = bytes_left < current_member_size
                       ? 0
                       : bytes_left - (current_member_size - offset_left);
        offset_left = 0;
      }
    }
    return result;
  }

  if (is_union_type(type))
  {
    /**
     * Unions are not nice, let's go through every member
     * and get the biggest one! And then use it directly
     *
     * @warning there is a semantic difference on this when
     * compared to c:@F@__memset_impl. While this function
     * will yield the same result as `clang` would, ESBMC
     * will handle the dereference (in the __memset_impl)
     * using the first member, which can lead to overflows.
     * See GitHub Issue #639
     *
     */
    expr2tc result = gen_zero(type);
    constant_union2t &data = to_constant_union2t(result);

    uint64_t union_total_size = type_byte_size(type).to_uint64();
    // Let's find a member with the biggest size
    size_t n = to_union_type(type).members.size();
    size_t selected_member_index = n;

    for (size_t i = 0; i < n; i++)
    {
      if (
        type_byte_size(to_union_type(type).members[i]).to_uint64() ==
        union_total_size)
      {
        selected_member_index = i;
        break;
      }
    }

    assert(selected_member_index < n);

    const irep_idt &name =
      to_union_type(type).member_names[selected_member_index];
    const type2tc &member_type =
      to_union_type(type).members[selected_member_index];
    expr2tc member = member2tc(member_type, src, name);

    data.init_field = name;
    data.datatype_members[0] =
      gen_value_by_byte(member_type, member, value, num_of_bytes, offset);
    return data.datatype_members[0] ? result : expr2tc();
  }

  // Found a primitive! Just apply the function
  return gen_byte_expression(type, src, value, num_of_bytes, offset);
}

expr2tc goto_symex_utils::gen_byte_memcpy(
  const expr2tc &src,
  const expr2tc &dst,
  const size_t num_of_bytes,
  const size_t src_offset,
  const size_t dst_offset)
{
  // Technically we already did all these checks before, this is just
  // an extra for DEBUG builds.
  assert(
    (src->type->get_width() - src_offset) >= num_of_bytes &&
    (dst->type->get_width() - dst_offset) >= num_of_bytes);

  if (is_pointer_type(src) || is_pointer_type(dst))
    return expr2tc();

  // TODO: Not sure how to deal with different types
  if (src->type != dst->type)
    return expr2tc();

  expr2tc src_mask = gen_zero(src->type);
  expr2tc dst_mask = gen_zero(dst->type);

  const expr2tc eight = constant_int2tc(dst->type, BigInt(8));
  const expr2tc one = constant_int2tc(dst->type, BigInt(1));

  for (unsigned i = 0; i < num_of_bytes; i++)
    for (int m = 0; m < 8; m++)
    {
      src_mask = shl2tc(dst->type, src_mask, one);
      src_mask = bitor2tc(dst->type, src_mask, one);
      dst_mask = shl2tc(dst->type, dst_mask, one);
      dst_mask = bitor2tc(dst->type, dst_mask, one);
    }

  for (unsigned i = 0; i < dst_offset; i++)
    dst_mask = shl2tc(dst->type, dst_mask, eight);

  dst_mask = bitnot2tc(dst->type, dst_mask);
  dst_mask = bitand2tc(dst->type, dst, dst_mask);

  for (unsigned i = 0; i < src_offset; i++)
    src_mask = shl2tc(dst->type, src_mask, eight);

  src_mask = bitand2tc(dst->type, src, src_mask);

  // When dst_offset > src_offset
  for (unsigned i = src_offset; i < dst_offset; i++)
    src_mask = shl2tc(dst->type, src_mask, eight);

  // When dst_offsett < src_offset
  for (unsigned i = dst_offset; i < src_offset; i++)
    src_mask = lshr2tc(dst->type, src_mask, eight);

  expr2tc result = bitor2tc(dst->type, dst_mask, src_mask);
  simplify(result);
  return result;
}

static inline expr2tc do_memcpy_expression(
  const expr2tc &dst,
  const size_t &dst_offset,
  const expr2tc &src,
  const size_t &src_offset,
  const size_t num_of_bytes)
{
  if (num_of_bytes == 0)
    return dst;

  // Short-circuit
  if (
    dst->type == src->type && !dst_offset && !src_offset &&
    type_byte_size(dst->type).to_uint64() == num_of_bytes)
    return src;

  if (
    is_array_type(src->type) || is_array_type(dst->type) ||
    is_struct_type(dst->type) || is_union_type(dst->type) ||
    is_struct_type(src->type) || is_union_type(src->type))
  {
    log_debug("memcpy", "Only primitives are supported for now");
    return expr2tc();
  }

  // gen_byte_memcpy asserts on get_width(), which throws
  // symbolic_type_excp for empty/code types. Defer to the C impl in that
  // case rather than crash — seen when a Python defaultdict factory is a
  // lambda whose call result has no resolved primitive width.
  if (
    is_empty_type(src->type) || is_empty_type(dst->type) ||
    is_code_type(src->type) || is_code_type(dst->type))
  {
    log_debug("memcpy", "Skipping primitive path for empty/code type");
    return expr2tc();
  }

  // Base-case. Primitives!
  return goto_symex_utils::gen_byte_memcpy(
    src, dst, num_of_bytes, src_offset, dst_offset);
}

static void offset_simplifier(expr2tc &e)
{
  simplify(e);
}

// Shared core for memcpy and memmove. The optimised path computes the new
// destination value from the *current* (pre-assignment) bytes of both objects
// and then assigns it, so overlapping regions are handled correctly — i.e. it
// already has memmove semantics. memcpy and memmove therefore differ only in
// the C fallback they bump to (@p bump_name) when the optimisation can't apply.
void goto_symext::intrinsic_memcpy_impl(
  reachability_treet &art,
  const code_function_call2t &func_call,
  const std::string &bump_name)
{
  assert(func_call.operands.size() == 3 && "Wrong memcpy/memmove signature");

  using namespace std::string_literals;

  if (options.get_bool_option("no-simplify"))
  {
    bump_call(func_call, bump_name);
    return;
  }

  const execution_statet &ex_state = art.get_cur_state();
  if (ex_state.cur_state->guard.is_false())
    return;

  expr2tc dst_arg = func_call.operands[0];
  expr2tc src_arg = func_call.operands[1];
  expr2tc n_arg = func_call.operands[2];

  // Three steps:
  // 1. Check if n_arg is constant;
  // 2. Compute all SRC addresses and memory checks
  // 3. Compute all DST addresses, memory check and compute operation result

  cur_state->rename(n_arg);
  if (!n_arg || is_symbol2t(n_arg))
  {
    bump_call(func_call, bump_name);
    return;
  }

  simplify(n_arg);
  if (!is_constant_int2t(n_arg))
  {
    bump_call(func_call, bump_name);
    return;
  }

  const unsigned long number_of_bytes = to_constant_int2t(n_arg).as_ulong();

  // Now grab all sources

  std::list<dereference_callbackt::internal_item> src_items;
  expr2tc src_deref = dereference2tc(get_empty_type(), src_arg);
  internal_deref_items.clear();
  dereference(src_deref, dereferencet::INTERNAL);

  if (!internal_deref_items.size())
  {
    bump_call(func_call, bump_name);
    return;
  }

  src_items.splice(src_items.end(), internal_deref_items);
  assert(internal_deref_items.size() == 0);

  // Sane checks here
  for (dereference_callbackt::internal_item &item : src_items)
  {
    guard2tc guard = ex_state.cur_state->guard;
    guard.add(item.guard);
    expr2tc &item_object = item.object;
    expr2tc &item_offset = item.offset;

    cur_state->rename(item_object);
    cur_state->rename(item_offset);

    if (!item_object || !item_offset)
    {
      bump_call(func_call, bump_name);
      return;
    }

    offset_simplifier(item_offset);
    if (!is_constant_int2t(item_offset))
    {
      bump_call(func_call, bump_name);
      return;
    }

    const uint64_t number_of_offset =
      to_constant_int2t(item_offset).value.to_uint64();

    uint64_t type_size;
    try
    {
      type_size = type_byte_size(item_object->type).to_uint64();
    }
    catch (const array_type2t::dyn_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }
    catch (const array_type2t::inf_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }

    if (is_code_type(item_object->type))
    {
      if (config.options.get_bool_option("enable-unreachability-intrinsic"))
      {
        // Workaround:
        // linux-3.10-rc1-43_1a-bitvector-drivers--net--ethernet--broadcom--b44.ko--ldv_main0.cil.out.i
        // generates an INVALID address pointing to both a struct and
        // initializes an extern global function ptr with. Resulting in this
        // being triggered wrongly. Need to check if it's a VSA issue or ESBMC
        // initialization issue.
        bump_call(func_call, bump_name);
        return;
      }

      std::string error_msg =
        fmt::format("dereference failure: trying to deref a ptr code");

      // SAME_OBJECT(ptr, item) => DEREF ERROR
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }

    // Over reading?
    bool is_out_bounds = ((type_size - number_of_offset) < number_of_bytes) ||
                         (number_of_offset > type_size);
    if (
      is_out_bounds && !options.get_bool_option("no-pointer-check") &&
      !options.get_bool_option("no-bounds-check"))
    {
      std::string error_msg = fmt::format(
        "dereference failure on memcpy: reading memory segment of size {} with "
        "{} "
        "bytes",
        type_size - number_of_offset,
        number_of_bytes);

      // SAME_OBJECT(ptr, item) => DEREF ERROR
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }
  }

  // Readings are sorted... now go for writings
  expr2tc dst_deref = dereference2tc(get_empty_type(), dst_arg);
  dereference(dst_deref, dereferencet::INTERNAL);

  for (dereference_callbackt::internal_item &item : internal_deref_items)
  {
    guard2tc guard = ex_state.cur_state->guard;
    guard.add(item.guard);
    // expr2tc &item_object = item.object;
    // expr2tc &item_offset = item.offset;

    cur_state->rename(item.guard);
    cur_state->rename(item.offset);

    offset_simplifier(item.offset);
    if (!is_constant_int2t(item.offset))
    {
      bump_call(func_call, bump_name);
      return;
    }

    const uint64_t number_of_offset =
      to_constant_int2t(item.offset).value.to_uint64();

    uint64_t type_size;
    try
    {
      type_size = type_byte_size(item.object->type).to_uint64();
    }
    catch (const array_type2t::dyn_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }
    catch (const array_type2t::inf_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }
    bool is_out_bounds = ((type_size - number_of_offset) < number_of_bytes) ||
                         (number_of_offset > type_size);
    if (
      is_out_bounds && !options.get_bool_option("no-pointer-check") &&
      !options.get_bool_option("no-bounds-check"))
    {
      std::string error_msg = fmt::format(
        "dereference failure on memcpy: writing memory segment of size {} with "
        "{} "
        "bytes",
        type_size - number_of_offset,
        number_of_bytes);

      // SAME_OBJECT(ptr, item) => DEREF ERROR
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }

    // Time to do the actual copy
    for (const auto &src_item : src_items)
    {
      // Offset is garanteed to be a constant
      const uint64_t src_offset =
        to_constant_int2t(src_item.offset).value.to_uint64();
      const expr2tc new_object = do_memcpy_expression(
        item.object,
        number_of_offset,
        src_item.object,
        src_offset,
        number_of_bytes);

      if (!new_object)
      {
        bump_call(func_call, bump_name);
        return;
      }

      guard2tc assignment_guard = guard;
      assignment_guard.add(src_item.guard);

      symex_assign(
        code_assign2tc(item.object, new_object), false, assignment_guard);
    }
  }
  if (!options.get_bool_option("no-pointer-check"))
  {
    expr2tc null_sym = symbol2tc(dst_arg->type, "NULL");

    expr2tc dst_same = same_object2tc(dst_arg, null_sym);
    expr2tc dst_null_check = not2tc(same_object2tc(dst_arg, null_sym));
    ex_state.cur_state->guard.guard_expr(dst_null_check);
    claim(dst_null_check, " dereference failure: NULL pointer on DST");

    expr2tc src_same = same_object2tc(src_arg, null_sym);
    expr2tc src_null_check = not2tc(same_object2tc(src_arg, null_sym));
    ex_state.cur_state->guard.guard_expr(src_null_check);
    claim(src_null_check, " dereference failure: NULL pointer on SRC");
  }

  expr2tc ret_ref = func_call.ret;
  if (!is_nil_expr(ret_ref))
  {
    dereference(ret_ref, dereferencet::READ);
    symex_assign(code_assign2tc(ret_ref, dst_arg), false, cur_state->guard);
  }
}

void goto_symext::intrinsic_memcpy(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  intrinsic_memcpy_impl(art, func_call, "c:@F@__memcpy_impl");
}

void goto_symext::intrinsic_memmove(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  // memmove is byte-identical to memcpy in the optimised path (the new value
  // is built from the current bytes of src/dst before assigning, so overlap is
  // handled); only the C fallback differs.
  intrinsic_memcpy_impl(art, func_call, "c:@F@__memmove_impl");
}

// Resolve @p ptr to a single concrete primitive object with a constant offset.
// Returns false (caller should bump to the C loop) if the pointer is symbolic,
// resolves to multiple objects, has a non-constant offset, or the n-byte read
// would run past the object. On success fills @p object / @p offset.
bool goto_symext::memcmp_resolve_operand(
  const expr2tc &ptr,
  unsigned long number_of_bytes,
  expr2tc &object,
  uint64_t &offset,
  uint64_t &avail_bytes)
{
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), ptr);
  dereference(deref, dereferencet::INTERNAL);

  // Exactly one target object — a single concrete primitive region.
  if (internal_deref_items.size() != 1)
    return false;

  dereference_callbackt::internal_item item = internal_deref_items.front();
  cur_state->rename(item.object);
  cur_state->rename(item.offset);
  if (!item.object || !item.offset)
    return false;

  offset_simplifier(item.offset);
  if (!is_constant_int2t(item.offset))
    return false;
  offset = to_constant_int2t(item.offset).value.to_uint64();

  // Scalars and fixed-size arrays are byte-extractable. Structs/unions have a
  // non-flat byte layout we don't model here, and code/empty types have no
  // width — defer those to the C loop. (Dynamically/infinitely sized arrays
  // throw from type_byte_size below and also fall back.)
  const type2tc &t = item.object->type;
  if (
    is_struct_type(t) || is_union_type(t) || is_code_type(t) ||
    is_empty_type(t))
    return false;

  uint64_t type_size;
  try
  {
    type_size = type_byte_size(t).to_uint64();
  }
  catch (const array_type2t::dyn_sized_array_excp &)
  {
    return false;
  }
  catch (const array_type2t::inf_sized_array_excp &)
  {
    return false;
  }

  if (offset > type_size)
    return false;
  // For a constant length, require the read to fit; for a symbolic length the
  // caller bounds the comparison by avail_bytes instead.
  if (number_of_bytes != 0 && (type_size - offset) < number_of_bytes)
    return false;

  object = item.object;
  avail_bytes = type_size - offset;
  return true;
}

void goto_symext::intrinsic_memcmp(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  assert(func_call.operands.size() == 3 && "Wrong memcmp signature");

  using namespace std::string_literals;
  const auto bump_name = "c:@F@__memcmp_impl"s;

  // --no-simplify keeps the literal C loop, matching memcpy/memset.
  if (options.get_bool_option("no-simplify"))
  {
    bump_call(func_call, bump_name);
    return;
  }

  const execution_statet &ex_state = art.get_cur_state();
  if (ex_state.cur_state->guard.is_false())
    return;

  expr2tc s1_arg = func_call.operands[0];
  expr2tc s2_arg = func_call.operands[1];
  expr2tc n_arg = func_call.operands[2];

  expr2tc ret_ref = func_call.ret;
  if (is_nil_expr(ret_ref))
    return; // result unused; nothing to model

  // Determine whether n is a known constant. A symbolic n is still handled
  // below, by bounding the comparison with the (concrete) object widths and
  // masking each byte position i with the predicate i < n.
  cur_state->rename(n_arg);
  if (!n_arg)
  {
    bump_call(func_call, bump_name);
    return;
  }
  simplify(n_arg);
  const bool n_is_const = is_constant_int2t(n_arg);
  const unsigned long const_n =
    n_is_const ? to_constant_int2t(n_arg).as_ulong() : 0;

  // n == 0 (constant): memcmp returns 0 without reading either pointer.
  if (n_is_const && const_n == 0)
  {
    dereference(ret_ref, dereferencet::READ);
    symex_assign(
      code_assign2tc(ret_ref, gen_zero(ret_ref->type)),
      false,
      cur_state->guard);
    return;
  }

  // Resolve both operands to concrete primitive objects. For a constant n the
  // resolver also checks the n-byte read is in bounds; for symbolic n it
  // reports the available byte count, which we use as the static comparison
  // bound.
  expr2tc obj1, obj2;
  uint64_t off1, off2, avail1, avail2;
  const unsigned long want = n_is_const ? const_n : 0;
  if (
    !memcmp_resolve_operand(s1_arg, want, obj1, off1, avail1) ||
    !memcmp_resolve_operand(s2_arg, want, obj2, off2, avail2))
  {
    bump_call(func_call, bump_name);
    return;
  }

  // Static byte bound for the unrolled comparison:
  //   constant n -> exactly n bytes (already checked in bounds);
  //   symbolic n -> min(avail1, avail2) bytes, with each position guarded by
  //                 i < n so positions at/after n contribute nothing.
  const uint64_t avail = std::min(avail1, avail2);
  const uint64_t nbytes = n_is_const ? const_n : avail;

  // Cap the unrolled width: a very large object would explode the ite chain.
  // Beyond this, defer to the C loop (which --unwind bounds anyway).
  static const uint64_t MAX_MEMCMP_UNROLL = 64;
  if (nbytes > MAX_MEMCMP_UNROLL)
  {
    bump_call(func_call, bump_name);
    return;
  }

  // Soundness for symbolic n: the unrolled comparison only examines the first
  // `avail` bytes (the smaller object's remaining size). If n could exceed
  // `avail`, the real memcmp would read past an object — an out-of-bounds
  // access we must not silently drop. Claim n <= avail so that path is
  // flagged as a dereference failure, exactly as the C loop's reads would be.
  if (
    !n_is_const && !options.get_bool_option("no-bounds-check") &&
    !options.get_bool_option("no-pointer-check"))
  {
    expr2tc in_bounds =
      lessthanequal2tc(n_arg, constant_int2tc(n_arg->type, BigInt(avail)));
    guard2tc g = ex_state.cur_state->guard;
    claim(
      implies2tc(g.as_expr(), in_bounds),
      "dereference failure: memcmp length exceeds object bounds");
  }

  // Build the lexicographic result as a nested ite over the byte reads, from
  // the last byte backwards so the first differing byte dominates:
  //   res = ite(active[0] && b1[0] != b2[0], (int)b1[0] - (int)b2[0],
  //         ite(active[1] && ..., ..., 0))
  // where active[i] is (i < n) for symbolic n, or always-true for constant n.
  // No loop is emitted, so nothing unwinds.
  const type2tc byte_t = get_uint_type(8);
  const type2tc int_t = signedbv_type2tc(config.ansi_c.int_width);
  const bool be = config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN;
  const type2tc n_t = n_arg->type;

  expr2tc res = gen_zero(int_t);
  for (long i = (long)nbytes - 1; i >= 0; --i)
  {
    expr2tc b1 =
      byte_extract2tc(byte_t, obj1, gen_ulong(off1 + (uint64_t)i), be);
    expr2tc b2 =
      byte_extract2tc(byte_t, obj2, gen_ulong(off2 + (uint64_t)i), be);
    expr2tc diff =
      sub2tc(int_t, typecast2tc(int_t, b1), typecast2tc(int_t, b2));
    expr2tc differs = notequal2tc(b1, b2);
    if (!n_is_const)
    {
      // byte i is examined only when i < n
      expr2tc within =
        lessthan2tc(constant_int2tc(n_t, BigInt((uint64_t)i)), n_arg);
      // also allow i == n boundary? memcmp compares indices [0, n), so i < n.
      differs = and2tc(within, differs);
    }
    res = if2tc(int_t, differs, diff, res);
  }

  dereference(ret_ref, dereferencet::READ);
  symex_assign(code_assign2tc(ret_ref, res), false, cur_state->guard);
}

/**
 * @brief Intrinsic for C memchr.
 *
 * memchr(const void *buf, int ch, size_t n) scans the first n bytes of the
 * object pointed to by buf for the first byte equal to (unsigned char)ch and
 * returns a pointer to it, or NULL if no such byte appears in those n bytes.
 *
 * Instead of unwinding the operational-model loop (one branch per byte), we
 * read the n candidate bytes with byte_extract and fold them into a single
 * nested-ite pointer expression:
 *
 *   res = ite(byte[0] == ch, buf + 0,
 *         ite(byte[1] == ch, buf + 1,
 *         ...
 *         ite(byte[n-1] == ch, buf + (n-1), NULL)))
 *
 * Building from the last position outward keeps the *first* match as the
 * outermost selected arm. ch is symbolic-friendly (the comparison is encoded);
 * only n must be a known constant. If the object/offset cannot be resolved to
 * constants we bump to the __memchr_impl loop.
 */
void goto_symext::intrinsic_memchr(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  assert(func_call.operands.size() == 3 && "Wrong memchr signature");

  using namespace std::string_literals;
  const auto bump_name = "c:@F@__memchr_impl"s;

  if (options.get_bool_option("no-simplify"))
  {
    bump_call(func_call, bump_name);
    return;
  }

  const execution_statet &ex_state = art.get_cur_state();
  if (ex_state.cur_state->guard.is_false())
    return;

  expr2tc buf_arg = func_call.operands[0];
  expr2tc ch_arg = func_call.operands[1];
  expr2tc n_arg = func_call.operands[2];

  cur_state->rename(ch_arg);
  cur_state->rename(n_arg);
  if (!ch_arg || !n_arg || is_symbol2t(n_arg))
  {
    bump_call(func_call, bump_name);
    return;
  }

  simplify(n_arg);
  if (!is_constant_int2t(n_arg))
  {
    bump_call(func_call, bump_name);
    return;
  }

  const unsigned long number_of_bytes = to_constant_int2t(n_arg).as_ulong();

  // Resolve the object(s) buf points to.
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), buf_arg);
  dereference(deref, dereferencet::INTERNAL);

  if (!internal_deref_items.size())
  {
    bump_call(func_call, bump_name);
    return;
  }

  std::list<dereference_callbackt::internal_item> buf_items;
  buf_items.splice(buf_items.end(), internal_deref_items);

  const bool is_big_endian =
    (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);

  expr2tc ret_ref = func_call.ret;
  if (is_nil_expr(ret_ref))
    return;

  const type2tc ret_type = ret_ref->type;
  const expr2tc null_result = symbol2tc(ret_type, "NULL");

  // Byte value to search for: (unsigned char)ch.
  const expr2tc ch_byte = typecast2tc(get_uint_type(8), ch_arg);

  for (auto &item : buf_items)
  {
    guard2tc guard = ex_state.cur_state->guard;
    guard.add(item.guard);

    expr2tc item_object = item.object;
    expr2tc item_offset = item.offset;
    cur_state->rename(item_object);
    cur_state->rename(item_offset);

    if (!item_object || !item_offset)
    {
      bump_call(func_call, bump_name);
      return;
    }

    offset_simplifier(item_offset);
    if (!is_constant_int2t(item_offset))
    {
      bump_call(func_call, bump_name);
      return;
    }

    const uint64_t number_of_offset =
      to_constant_int2t(item_offset).value.to_uint64();

    if (is_code_type(item_object->type))
    {
      bump_call(func_call, bump_name);
      return;
    }

    uint64_t type_size;
    try
    {
      type_size = type_byte_size(item_object->type).to_uint64();
    }
    catch (const array_type2t::dyn_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }
    catch (const array_type2t::inf_sized_array_excp &)
    {
      bump_call(func_call, bump_name);
      return;
    }

    // Reading the first n bytes must stay within the object.
    const bool is_out_bounds =
      (number_of_offset > type_size) ||
      ((type_size - number_of_offset) < number_of_bytes);
    if (
      is_out_bounds && !options.get_bool_option("no-pointer-check") &&
      !options.get_bool_option("no-bounds-check"))
    {
      std::string error_msg = fmt::format(
        "dereference failure on memchr: reading memory segment of size {} with "
        "{} bytes",
        type_size - number_of_offset,
        number_of_bytes);
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }

    // Fold the n candidate bytes into a nested-ite pointer, from the last
    // position outward so the first match wins.
    expr2tc result = null_result;
    for (unsigned long k = number_of_bytes; k-- > 0;)
    {
      expr2tc byte = byte_extract2tc(
        get_uint_type(8),
        item_object,
        gen_ulong(number_of_offset + k),
        is_big_endian);
      expr2tc match = equality2tc(byte, ch_byte);
      expr2tc hit = add2tc(ret_type, buf_arg, gen_ulong(k));
      result = if2tc(ret_type, match, hit, result);
    }

    symex_assign(code_assign2tc(ret_ref, result), false, guard);
  }

  // C11/C17 7.24.5.1: with n == 0 no bytes are examined, so buf is never
  // dereferenced and need not be valid. The __memchr_impl model agrees
  // (its loop is `while (n && ...)`). Only claim non-NULL when n > 0.
  if (number_of_bytes != 0 && !options.get_bool_option("no-pointer-check"))
  {
    expr2tc null_sym = symbol2tc(buf_arg->type, "NULL");
    expr2tc null_check = not2tc(same_object2tc(buf_arg, null_sym));
    ex_state.cur_state->guard.guard_expr(null_check);
    claim(null_check, " dereference failure: NULL pointer on memchr");
  }
}

/**
 * @brief This function will try to initialize the object pointed by
 * the address in a smarter way, minimizing the number of assignments.
 * This is intend to optimize the behavior of a memset operation:
 *
 * memset(void* ptr, int value, size_t num_of_bytes)
 *
 * - ptr can point to anything. We have to add checks!
 * - value is interpreted as a uchar.
 * - num_of_bytes must be known. If it is nondet, we will bump the call
 *
 * In plain C, the objective of a call such as:
 *
 * int a;
 * memset(&a, value, num)
 *
 * Would generate something as:
 *
 * int temp = 0;
 * for(int i = 0; i < num; i++) temp = byte | (temp << 8);
 * a = temp;
 *
 * This is just a simplification for understanding though. During the
 * instrumentation size checks will be added, and also, the original
 * bytes from `a` that were not overwritten must be mantained!
 * Arrays will need to be added up to an nth element.
 *
 * In ESBMC though, we have 2 main methods of dealing with memory objects:
 *
 * A. Heap objects, which are valid/invalid. They are the easiest to deal
 *    with, as the dereference will actually return a big array of char to us.
 *    For this case, we can just overwrite the members directly with the value
 *
 * B. Stack objects, which are typed. It will be hard, this will require operations
 *    which depends on the base type and also on padding.
 */
void goto_symext::intrinsic_memset(
  reachability_treet &art,
  const code_function_call2t &func_call)
{
  // 1. Check for the functions parameters and do the deref and processing!

  assert(func_call.operands.size() == 3 && "Wrong memset signature");
  const execution_statet &ex_state = art.get_cur_state();
  if (ex_state.cur_state->guard.is_false())
    return;

  /* Get the arguments
   * arg0: ptr to object
   * arg1: int for the new byte value
   * arg2: number of bytes to be set */
  expr2tc arg0 = func_call.operands[0];
  expr2tc arg1 = func_call.operands[1];
  expr2tc arg2 = func_call.operands[2];

  // Checks where arg0 points to
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), arg0);
  dereference(deref, dereferencet::INTERNAL);

  /* Preconditions for the optimization:
   * A: It should point to someplace
   * B: byte itself should be renamed properly
   * C: Number of bytes cannot be symbolic
   * D: This is a simplification. So don't run with --no-simplify */
  cur_state->rename(arg1);
  cur_state->rename(arg2);
  if (
    !internal_deref_items.size() || !arg1 || !arg2 || is_symbol2t(arg2) ||
    options.get_bool_option("no-simplify"))
  {
    /* Not sure what to do here, let's rely
       * on the default implementation then */
    log_debug("memset", "Couldn't optimize memset due to precondition");
    bump_call(func_call, "c:@F@__memset_impl");
    return;
  }

  simplify(arg2);
  if (!is_constant_int2t(arg2))
  {
    log_debug("memset", "TODO: simplifier issues :/");
    bump_call(func_call, "c:@F@__memset_impl");
    return;
  }

  unsigned long number_of_bytes = to_constant_int2t(arg2).as_ulong();

  // If any potential target is read-only (string literal or const global/static),
  // fall back to __memset_impl, which uses WRITE-mode dereferences and reports
  // the proper violation via valid_check() in dereference.cpp.
  for (const auto &item : internal_deref_items)
  {
    const expr2tc *base = &item.object;
    while (is_member2t(*base))
      base = &to_member2t(*base).source_value;
    while (is_index2t(*base))
      base = &to_index2t(*base).source_value;
    if (is_constant_string2t(*base))
    {
      bump_call(func_call, "c:@F@__memset_impl");
      return;
    }
    if (is_symbol2t(*base))
    {
      const symbolt *sym = ns.lookup(to_symbol2t(*base).thename);
      if (
        sym != nullptr && sym->static_lifetime &&
        sym->get_type().cmt_constant())
      {
        bump_call(func_call, "c:@F@__memset_impl");
        return;
      }
    }
  }

  // Where are we pointing to?
  for (auto &item : internal_deref_items)
  {
    guard2tc guard = ex_state.cur_state->guard;
    expr2tc item_object = item.object;
    expr2tc item_offset = item.offset;
    guard.add(item.guard);

    cur_state->rename(item_object);
    cur_state->rename(item_offset);

    /* Pre-requisites locally:
       * item_object must be something!
       * item_offset must be something! */
    if (!item_object || !item_offset)
    {
      log_debug("memset", "Couldn't get item_object/item_offset");
      bump_call(func_call, "c:@F@__memset_impl");
      return;
    }

    simplify(item_offset);

    if (!is_constant_int2t(item_offset))
    {
      /* If we reached here, item_offset is not symbolic
       * and we don't know what the actual value of it is...
       *
       * For now bump_call, later we should expand our simplifier
       */
      log_debug(
        "memset", "TODO: some simplifications are missing, bumping call");
      bump_call(func_call, "c:@F@__memset_impl");
      return;
    }

    uint64_t number_of_offset =
      to_constant_int2t(item_offset).value.to_uint64();

    /* This fails for VLAs or dynamically allocated arrays.
     * XXX: We could consider not failing and encoding the is_out_bounds
     *      condition below symbolically instead. */
    uint64_t type_size;
    try
    {
      type_size = type_byte_size(item_object->type).to_uint64();
    }
    catch (const array_type2t::dyn_sized_array_excp &)
    {
      bump_call(func_call, "c:@F@__memset_impl");
      return;
    }

    if (is_code_type(item_object->type))
    {
      std::string error_msg =
        fmt::format("dereference failure: trying to deref a ptr code");

      // SAME_OBJECT(ptr, item) => DEREF ERROR
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }

    bool is_out_bounds = ((type_size - number_of_offset) < number_of_bytes) ||
                         (number_of_offset > type_size);
    if (
      is_out_bounds && !options.get_bool_option("no-pointer-check") &&
      !options.get_bool_option("no-bounds-check"))
    {
      std::string error_msg = fmt::format(
        "dereference failure: memset of memory segment of size {} with {} "
        "bytes",
        type_size - number_of_offset,
        number_of_bytes);

      // SAME_OBJECT(ptr, item) => DEREF ERROR
      expr2tc check = implies2tc(item.guard, gen_false_expr());
      claim(check, error_msg);
      continue;
    }

    expr2tc new_object = gen_value_by_byte(
      item_object->type, item_object, arg1, number_of_bytes, number_of_offset);

    // Were we able to optimize it? If not... bump call
    if (!new_object)
    {
      log_debug("memset", "gen_value_by_byte failed");
      bump_call(func_call, "c:@F@__memset_impl");
      return;
    }
    // 4. Assign the new object
    symex_assign(code_assign2tc(item.object, new_object), false, guard);
  }
  // Lastly, let's add a NULL ptr check
  if (!options.get_bool_option("no-pointer-check"))
  {
    expr2tc null_sym = symbol2tc(arg0->type, "NULL");
    expr2tc obj = same_object2tc(arg0, null_sym);
    expr2tc null_check = not2tc(same_object2tc(arg0, null_sym));
    ex_state.cur_state->guard.guard_expr(null_check);
    claim(null_check, " dereference failure: NULL pointer");
  }

  expr2tc ret_ref = func_call.ret;
  if (!is_nil_expr(ret_ref))
  {
    dereference(ret_ref, dereferencet::READ);
    symex_assign(code_assign2tc(ret_ref, arg0), false, cur_state->guard);
  }
}
