#include <cassert>
#include <goto-symex/goto_symex.h>
#include <goto-symex/reachability_tree.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/std_types.h>

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

  // Helper lambda for creating fallback size values.
  // GCC's __builtin_object_size returns:
  //   - (size_t)-1 if the object cannot be determined (for type=0 or 1),
  //   - 0 if the object cannot be determined (for type=2 or 3).
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
    // Unable to determine the underlying object.
    // Fall back to GCC semantics depending on type:
    //   type 0/1 → (size_t)-1
    //   type 2/3 → 0
    obj_size = create_fallback_size(use_zero_for_unknown);
  }
  else
  {
    type2tc addressed_type;

    // Determine addressed type from address_of expressions
    if (is_address_of2t(ptr))
    {
      const address_of2t &addrof = to_address_of2t(ptr);
      if (is_index2t(addrof.ptr_obj))
      {
        const index2t &idx = to_index2t(addrof.ptr_obj);
        if (is_symbol2t(idx.source_value) || is_member2t(idx.source_value))
          addressed_type = idx.source_value->type;
      }
      else if (is_member2t(addrof.ptr_obj) || is_symbol2t(addrof.ptr_obj))
        addressed_type = addrof.ptr_obj->type;
    }

    // Handle nil addressed type cases
    if (is_nil_type(addressed_type))
    {
      if (is_pointer_type(ptr->type))
      {
        type2tc ptr_subtype = to_pointer_type(ptr->type).subtype;
        const auto &item = internal_deref_items.front();

        if (
          is_constant_int2t(item.offset) && is_struct_type(item.object->type) &&
          !is_nil_expr(deref) && !is_empty_type(deref->type))
        {
          addressed_type = deref->type;
        }

        if (is_nil_type(addressed_type))
        {
          if (is_symbol_type(ptr_subtype))
          {
            const symbol_type2t &symtype = to_symbol_type(ptr_subtype);
            const symbolt *symbol = ns.lookup(symtype.symbol_name);
            addressed_type = (symbol != nullptr)
                               ? migrate_type(symbol->type)
                               : internal_deref_items.front().object->type;
          }
          else
          {
            addressed_type =
              is_array_type(internal_deref_items.front().object->type)
                ? internal_deref_items.front().object->type
                : ptr_subtype;
          }
        }
      }
      else
        addressed_type = internal_deref_items.front().object->type;
    }

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

  assert(is_array_type(internal_deref_items.front().object->type));
  expr2tc obj_size =
    to_array_type(internal_deref_items.front().object->type).array_size;

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
