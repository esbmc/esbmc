#include <cassert>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/irep/std_types.h>

namespace
{
/// The object an `address_of` names directly, or nil when `ptr` is not one or
/// does not name an object whose type is the whole object's.
type2tc address_of_object_type(const expr2tc &ptr)
{
  if (!is_address_of2t(ptr))
    return type2tc();

  const address_of2t &addrof = to_address_of2t(ptr);
  if (is_index2t(addrof.ptr_obj))
  {
    const index2t &idx = to_index2t(addrof.ptr_obj);
    if (is_symbol2t(idx.source_value) || is_member2t(idx.source_value))
      return idx.source_value->type;
    return type2tc();
  }

  if (is_member2t(addrof.ptr_obj) || is_symbol2t(addrof.ptr_obj))
    return addrof.ptr_obj->type;

  return type2tc();
}

/// The type of the object `ptr` addresses, for type-0 object-size purposes.
///
/// `deref_items` is the non-empty resolution of `ptr`; `deref` is the
/// dereference expression, consulted for a struct at a constant offset.
type2tc addressed_object_type(
  const expr2tc &ptr,
  const expr2tc &deref,
  const std::list<dereference_callbackt::internal_item> &deref_items,
  const namespacet &ns)
{
  if (type2tc named = address_of_object_type(ptr); !is_nil_type(named))
    return named;

  const type2tc &resolved = deref_items.front().object->type;
  if (!is_pointer_type(ptr->type))
    return resolved;

  const type2tc ptr_subtype = to_pointer_type(ptr->type).subtype;
  const auto &item = deref_items.front();

  if (
    is_constant_int2t(item.offset) && is_struct_type(item.object->type) &&
    !is_nil_expr(deref) && !is_empty_type(deref->type))
    return deref->type;

  if (is_symbol_type(ptr_subtype))
  {
    const symbol_type2t &symtype = to_symbol_type(ptr_subtype);
    const symbolt *symbol = ns.lookup(symtype.symbol_name);
    return symbol != nullptr ? migrate_symbol_type(*symbol) : resolved;
  }

  // A void* carries no size, so when the pointer's subtype is empty the
  // resolved object is the only thing that knows how big it is. Falling back to
  // the subtype there made every scalar reached through a void* report the
  // unknown-size fallback -- visible as __CPROVER_OBJECT_SIZE(&scalar_static)
  // failing where CBMC proves it, since the CPROVER lowering always casts to
  // void* first.
  return (is_array_type(resolved) || is_empty_type(ptr_subtype)) ? resolved
                                                                 : ptr_subtype;
}
} // namespace

void goto_symext::intrinsic_builtin_object_size(
  const code_function_call2t &func_call,
  reachability_treet &)
{
  assert(
    func_call.operands.size() == 2 && "Wrong __builtin_object_size signature");
  expr2tc ptr = func_call.operands[0];
  expr2tc type_param = func_call.operands[1];

  // Extract type parameter
  size_t type_value = 0;
  cur_state->rename(type_param);
  if (is_constant_int2t(type_param))
  {
    int64_t param_val = to_constant_int2t(type_param).value.to_int64();
    // Invalid type parameter (outside 0..3): keep default 0 (GCC behavior).
    if (param_val >= 0 && param_val <= 3)
      type_value = static_cast<size_t>(param_val);
  }

  // Work out what the ptr points at.
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), ptr);
  dereference(deref, dereferencet::INTERNAL);

  bool use_zero_for_unknown = (type_value == 2 || type_value == 3);
  bool consider_offset = (type_value == 1 || type_value == 3);

  // Helper lambda for creating fallback size values when the object
  // cannot be determined:
  //   - type 0/1: an SSIZE_MAX-style cap, 2^(word_size-1)-1 (GCC itself
  //     returns (size_t)-1),
  //   - type 2/3: 0.
  // The type parameter encodes whether we want the full size (0/2)
  // or remaining size after pointer offset (1/3).
  auto create_fallback_size = [&](bool use_zero) {
    return use_zero ? constant_int2tc(size_type2(), BigInt(0))
                    : constant_int2tc(
                        size_type2(),
                        BigInt((1ULL << (config.ansi_c.word_size - 1)) - 1));
  };

  expr2tc obj_size;

  if (internal_deref_items.empty())
  {
    // Unable to determine the underlying object; use the fallback sizes
    // described above.
    obj_size = create_fallback_size(use_zero_for_unknown);
  }
  else
  {
    const type2tc addressed_type =
      addressed_object_type(ptr, deref, internal_deref_items, ns);

    // Note: type_byte_size returns the allocated object size, not just the sum
    // of fields. For structs/unions this includes alignment and padding, which
    // matches GCC's __builtin_object_size semantics.
    BigInt total_size = type_byte_size(addressed_type);

    if (consider_offset)
    {
      // Type 1 or 3: calculate remaining bytes from offset
      expr2tc offset_expr = pointer_offset2tc(get_int64_type(), ptr);
      cur_state->rename(offset_expr);
      do_simplify(offset_expr);

      if (is_constant_int2t(offset_expr))
      {
        BigInt offset = to_constant_int2t(offset_expr).value;
        BigInt remaining =
          (total_size > offset) ? (total_size - offset) : BigInt(0);
        obj_size = constant_int2tc(size_type2(), remaining);
      }
      else
      {
        // Offset is symbolic - can't determine remaining size statically
        const expr2tc total_size_expr =
          constant_int2tc(get_int64_type(), total_size);
        obj_size = if2tc(
          size_type2(),
          greaterthan2tc(total_size_expr, offset_expr),
          sub2tc(size_type2(), total_size_expr, offset_expr),
          gen_zero(size_type2()));
      }
    }
    else
    {
      // Type 0 or 2: return full object size of the addressed object
      obj_size = constant_int2tc(size_type2(), total_size);
    }
  }

  expr2tc ret_ref = func_call.ret;
  if (!is_nil_expr(ret_ref))
  {
    dereference(ret_ref, dereferencet::READ);
    symex_assign(
      code_assign2tc(ret_ref, typecast2tc(ret_ref->type, obj_size)),
      false,
      cur_state->guard);
  }
}

void goto_symext::intrinsic_get_object_size(
  const code_function_call2t &func_call,
  reachability_treet &)
{
  assert(func_call.operands.size() == 1 && "Wrong get_object_size signature");
  expr2tc ptr = func_call.operands[0];

  // Work out what the ptr points at.
  internal_deref_items.clear();
  expr2tc deref = dereference2tc(get_empty_type(), ptr);
  dereference(deref, dereferencet::INTERNAL);

  // __ESBMC_get_object_size returns the element count of the array object the
  // pointer addresses. If the pointer cannot be resolved to a concrete object,
  // or the resolved object is not an array, that count is undefined: reading
  // internal_deref_items.front() on an empty container is UB (SIGSEGV in
  // release) and to_array_type() on a non-array trips an assertion. The Python
  // set/graph/bytes operational models can route such a pointer here (issues
  // #4782, #4804, #4805, #5658) — e.g. a `bytes` function parameter has no
  // compile-time array bound, so it decays to a plain pointer with nothing for
  // internal_deref_items to resolve. Rather than aborting, model the count as
  // an unconstrained non-negative nondet value: any concrete length the caller
  // could have passed is covered by some assignment, matching how symex
  // already treats other unresolvable-but-legal sizes (e.g. asprintf's
  // unbounded %s, io.cpp). The array path below is byte-for-byte unchanged, so
  // C/C++ callers — which always pass an array object — are unaffected.
  expr2tc obj_size;

  if (
    internal_deref_items.empty() ||
    !is_array_type(internal_deref_items.front().object->type))
  {
    log_debug(
      "goto-symex",
      "__ESBMC_get_object_size: object is not a resolvable array; "
      "modelling its size as an unconstrained nondet value");

    obj_size = sideeffect2tc(
      size_type2(),
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::allockind::nondet);
    replace_nondet(obj_size);
  }
  else
  {
    const type2tc &obj_type = internal_deref_items.front().object->type;
    obj_size = to_array_type(obj_type).array_size;
  }

  expr2tc ret_ref = func_call.ret;
  if (!is_nil_expr(ret_ref))
  {
    dereference(ret_ref, dereferencet::READ);
    symex_assign(
      code_assign2tc(ret_ref, typecast2tc(ret_ref->type, obj_size)),
      false,
      cur_state->guard);
  }
}
