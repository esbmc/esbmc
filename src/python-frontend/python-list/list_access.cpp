#include "python_list_internal.h"

using namespace python_expr;
using namespace python_list_detail;

exprt python_list::build_list_at_call(
  const exprt &list,
  const exprt &index,
  const nlohmann::json &element)
{
  pointer_typet obj_type(converter_.get_type_handler().get_list_element_type());

  const symbolt *list_at_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  assert(list_at_func_sym);

  locationt location = converter_.get_location_from_decl(element);

  // Get list size
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  assert(size_func && "list_size function not found");

  symbolt &size_var = converter_.create_tmp_symbol(
    element, "$list_size$", size_type(), gen_zero(size_type()));
  code_declt size_decl(build_symbol(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = build_symbol(*size_func);
  size_call.arguments().push_back(
    list.type().is_pointer() ? list : build_address_of(list));
  size_call.lhs() = build_symbol(size_var);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Convert index to size_t for comparison and arithmetic
  exprt index_as_size = build_typecast(index, size_type());

  // Create: actual_index = (index < 0) ? (size + index) : index
  // (V.3: build the negative-index normalization in IREP2.)
  const type2tc size_t2 = migrate_type(size_type());
  expr2tc index2, index_as_size2, size_var2;
  migrate_expr(index, index2);
  migrate_expr(index_as_size, index_as_size2);
  migrate_expr(build_symbol(size_var), size_var2);

  expr2tc is_negative = lessthan2tc(index2, gen_zero(index2->type));
  // For negative: size + index (since index is negative, this is
  // size - abs(index)).
  expr2tc positive_index = add2tc(size_t2, size_var2, index_as_size2);
  // Choose between positive conversion or original.
  expr2tc converted_index2 =
    if2tc(size_t2, is_negative, positive_index, index_as_size2);
  exprt converted_index = migrate_expr_back(converted_index2);

  if (!config.options.get_bool_option("no-bounds-check"))
  {
    // Runtime guard only for negative-index normalization. This prevents
    // underflowed indices (e.g., [] [-1]) from reaching the backend while
    // preserving legacy behavior for non-negative accesses.
    // negative_oob = is_negative && (converted_index >= size) (V.3: IREP2).
    expr2tc negative_oob =
      and2tc(is_negative, greaterthanequal2tc(converted_index2, size_var2));

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "list index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = migrate_expr_back(negative_oob);
    guard.then_case() = throw_code;
    guard.location() = location;
    converter_.add_instruction(guard);
  }

  // Use the converted expression directly in the call
  exprt list_at_call = build_call_expr(
    *list_at_func_sym,
    obj_type,
    {list.type().is_pointer() ? list : build_address_of(list),
     converted_index});
  list_at_call.location() = location;

  return list_at_call;
}

exprt python_list::index(const exprt &array, const nlohmann::json &slice_node)
{
  if (slice_node["_type"] == "Slice") // arr[lower:upper]
  {
    return handle_range_slice(array, slice_node);
  }
  else
  {
    return handle_index_access(array, slice_node);
  }
  return exprt();
}

exprt python_list::remove_function_calls_recursive(
  exprt &e,
  const nlohmann::json &node)
{
  // Bounds might generate intermediate calls, we need to add lhs to all of them.
  const auto add_lhs_var_bound = [&](exprt &foo) -> exprt {
    if (!foo.is_function_call())
      return foo;
    code_function_callt &call = static_cast<code_function_callt &>(foo);
    symbolt &lhs = converter_.create_tmp_symbol(
      node, "__python_function_call_lhs$", size_type(), exprt());
    call.lhs() = build_symbol(lhs);
    converter_.add_instruction(call);
    return build_symbol(lhs);
  };

  auto res = add_lhs_var_bound(e);
  for (auto &ee : res.operands())
  {
    ee = add_lhs_var_bound(ee);
    remove_function_calls_recursive(ee, node);
  }

  return res;
}

const symbolt &python_list::get_str_slice_sym()
{
  const std::string id = "c:@F@__python_str_slice";
  symbolt *sym = converter_.symbol_table().find_symbol(id);
  if (!sym)
  {
    symbolt new_symbol;
    new_symbol.name = "__python_str_slice";
    new_symbol.id = id;
    new_symbol.mode = "C";
    new_symbol.is_extern = true;
    // char* __python_str_slice(const char*, long long, long long, long long)
    code_typet slice_type;
    typet char_ptr = gen_pointer_type(char_type());
    typet ll_type = signedbv_typet(64);
    slice_type.return_type() = char_ptr;
    slice_type.arguments().push_back(code_typet::argumentt(char_ptr));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    slice_type.arguments().push_back(code_typet::argumentt(ll_type));
    new_symbol.set_type(slice_type);
    converter_.symbol_table().add(new_symbol);
    sym = converter_.symbol_table().find_symbol(id);
  }
  return *sym;
}

exprt python_list::handle_range_slice(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_array_type = ns.follow(array.type());
  const typet resolved_list_type = ns.follow(list_type);

  // Handle regular array/string slicing (not list slicing)
  // String parameters come as pointer-to-char, so handle both arrays and char pointers
  bool is_string_slice = (resolved_array_type != resolved_list_type &&
                          resolved_array_type.is_array()) ||
                         (resolved_array_type.is_pointer() &&
                          resolved_array_type.subtype() == char_type());

  // Determine step value (default 1).
  bool has_step = slice_node.contains("step") && !slice_node["step"].is_null();
  long long step_val = 1;
  bool literal_zero_step = false;
  if (has_step)
  {
    const auto &step_node = slice_node["step"];
    if (step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub")
    {
      step_val = -(long long)step_node["operand"]["value"].get<std::int64_t>();
    }
    else if (step_node["_type"] == "Constant")
    {
      step_val = step_node["value"].get<std::int64_t>();
    }
    if (step_val == 0)
    {
      literal_zero_step = true;
      step_val = 1; // continue with valid value to keep IR consistent
    }
  }
  // Python raises ValueError on step==0; emit a failing assertion so
  // verification reports it rather than silently producing a slice.
  // code_assertt does not insert assume(false), so the rest of the slice
  // IR still runs with step_val==1 — that is harmless: the violation is
  // already reported by the checker.
  if (literal_zero_step)
  {
    // V.3: build the always-fail assert condition in IREP2.
    code_assertt step_assert(migrate_expr_back(gen_false_expr()));
    step_assert.location() = converter_.get_location_from_decl(slice_node);
    step_assert.location().comment("ValueError: slice step cannot be zero");
    converter_.add_instruction(step_assert);
  }
  bool negative_step = (step_val < 0);

  if (is_string_slice)
  {
    locationt location = converter_.get_location_from_decl(slice_node);

    // For pointer types (function parameters), delegate to __python_str_slice
    // which uses __ESBMC_alloca and survives function returns.
    if (
      resolved_array_type.is_pointer() &&
      resolved_array_type.subtype() == char_type())
    {
      const symbolt &slice_func_ref = get_str_slice_sym();

      // Extract start/end bounds from slice node
      auto get_bound_expr =
        [&](const std::string &name, long long default_val) -> exprt {
        if (!slice_node.contains(name) || slice_node[name].is_null())
          return from_integer(default_val, signedbv_typet(64));

        const auto &bound = slice_node[name];
        if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
        {
          exprt abs_value = converter_.get_expr(bound["operand"]);
          // 0 - (int64)abs_value: both operands are signedbv64 (same width).
          return build_sub(
            from_integer(0, signedbv_typet(64)),
            build_typecast(abs_value, signedbv_typet(64)),
            signedbv_typet(64));
        }
        exprt e = converter_.get_expr(bound);
        e = remove_function_calls_recursive(e, slice_node);
        return build_typecast(e, signedbv_typet(64));
      };

      // Defaults: for positive step start=0,end=MAX; for negative step start=MAX,end=MIN
      // We use large sentinel values; __python_str_slice clamps them
      long long start_default = negative_step ? 999999 : 0;
      long long end_default = negative_step ? -999999 : 999999;

      exprt start_expr = get_bound_expr("lower", start_default);
      exprt end_expr = get_bound_expr("upper", end_default);
      exprt step_expr = from_integer(step_val, signedbv_typet(64));

      // Call __python_str_slice(s, start, end, step) as side-effect expression
      exprt slice_call = build_call_expr(
        slice_func_ref,
        pointer_typet(char_type()),
        {array, start_expr, end_expr, step_expr});
      slice_call.location() = location;

      return slice_call;
    }

    // For array types (local string literals), generate inline loop
    auto to_size_expr = [&](const exprt &expr) -> exprt {
      if (expr.type() == size_type())
        return expr;
      return build_typecast(expr, size_type());
    };
    auto size_add = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      plus_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_sub = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      minus_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_mul = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      mult_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_div = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      div_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };

    // Determine element type and logical length
    typet elem_type;
    exprt array_len;
    exprt logical_len;

    {
      const array_typet &src_type = to_array_type(resolved_array_type);
      elem_type = src_type.subtype();
      array_len = to_size_expr(src_type.size());
      // For char arrays (strings), exclude null terminator from logical length
      logical_len = (elem_type == char_type())
                      ? size_sub(array_len, gen_one(size_type()))
                      : array_len;
    }

    // Process slice bounds (handles null, negative indices)
    auto process_bound =
      [&](const std::string &bound_name, const exprt &default_value) -> exprt {
      if (!slice_node.contains(bound_name) || slice_node[bound_name].is_null())
        return to_size_expr(default_value);

      const auto &bound = slice_node[bound_name];

      // Check if negative index
      if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
      {
        exprt abs_value = to_size_expr(converter_.get_expr(bound["operand"]));
        // Clamp to 0 when abs_value > logical_len (avoids unsigned underflow).
        // (V.3: built in IREP2.)
        const type2tc size_t2 = migrate_type(size_type());
        expr2tc abs2, len2, converted2;
        migrate_expr(abs_value, abs2);
        migrate_expr(logical_len, len2);
        migrate_expr(size_sub(logical_len, abs_value), converted2);
        return migrate_expr_back(if2tc(
          size_t2, greaterthan2tc(abs2, len2), gen_zero(size_t2), converted2));
      }

      exprt e = converter_.get_expr(bound);
      return to_size_expr(remove_function_calls_recursive(e, slice_node));
    };

    // Process bounds: defaults depend on step direction
    exprt lower_expr, upper_expr;
    if (negative_step)
    {
      // For negative step: default lower = len-1, default upper = -1 (before 0)
      lower_expr =
        process_bound("lower", size_sub(logical_len, gen_one(size_type())));
    }
    else
    {
      lower_expr = process_bound("lower", gen_zero(size_type()));
    }

    // Upper bound
    if (!negative_step)
      upper_expr = process_bound("upper", logical_len);

    // Clamp bounds to [0, logical_len] to match Python semantics.
    if (!negative_step)
    {
      // bound = (bound >= logical_len) ? logical_len : bound
      // (V.3: built in IREP2.)
      const type2tc size_t2 = migrate_type(size_type());
      expr2tc len2;
      migrate_expr(logical_len, len2);
      auto clamp_to_len = [&](exprt &bound) {
        expr2tc b2;
        migrate_expr(bound, b2);
        bound = migrate_expr_back(
          if2tc(size_t2, greaterthanequal2tc(b2, len2), len2, b2));
      };
      clamp_to_len(lower_expr);
      clamp_to_len(upper_expr);
    }

    // Calculate slice length
    exprt slice_len;
    if (negative_step)
    {
      // For [::-1]: length = lower + 1 (e.g., lower=len-1 → length=len)
      if (
        (!slice_node.contains("lower") || slice_node["lower"].is_null()) &&
        (!slice_node.contains("upper") || slice_node["upper"].is_null()))
      {
        slice_len = logical_len;
      }
      else
      {
        slice_len = size_add(lower_expr, gen_one(size_type()));
      }
    }
    else if (step_val != 1)
    {
      // For step > 1: length = ceil((upper - lower) / step)
      exprt range =
        size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
      exprt step_const = from_integer(step_val, size_type());
      exprt step_minus_one = from_integer(step_val - 1, size_type());
      slice_len = size_div(size_add(range, step_minus_one), step_const);
    }
    else
    {
      slice_len = size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
    }

    // Create result array type with extra space for null terminator
    plus_exprt result_size(slice_len, gen_one(size_type()));
    result_size.type() = size_type();
    array_typet result_type(elem_type, result_size);

    // Create temporary for sliced array
    symbolt &result = converter_.create_tmp_symbol(
      slice_node, "$array_slice$", result_type, exprt());

    code_declt result_decl(build_symbol(result));
    result_decl.location() = location;
    converter_.add_instruction(result_decl);

    // Create loop: for i = 0; i < slice_len; i++
    symbolt &idx = converter_.create_tmp_symbol(
      slice_node, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(build_symbol(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    // idx is size_type; slice_len is built via size_add/sub/div (size_type),
    // so both operands share width.
    exprt cond = build_less_than(build_symbol(idx), slice_len);

    code_blockt body;

    // Compute source index based on step direction
    exprt src_idx;
    if (negative_step)
    {
      // result[i] = array[lower - i] (for step=-1)
      // For other negative steps: result[i] = array[lower - i * |step|]
      if (step_val == -1)
        src_idx = size_sub(to_size_expr(lower_expr), build_symbol(idx));
      else
      {
        exprt abs_step = from_integer(-step_val, size_type());
        src_idx = size_sub(
          to_size_expr(lower_expr), size_mul(build_symbol(idx), abs_step));
      }
    }
    else if (step_val != 1)
    {
      // result[i] = array[lower + i * step]
      exprt step_const = from_integer(step_val, size_type());
      src_idx = size_add(
        to_size_expr(lower_expr), size_mul(build_symbol(idx), step_const));
    }
    else
    {
      // result[i] = array[lower + i]
      src_idx = size_add(to_size_expr(lower_expr), build_symbol(idx));
    }

    exprt src = build_index(array, src_idx, elem_type);
    exprt dst = build_index(build_symbol(result), build_symbol(idx), elem_type);
    code_assignt assign(dst, src);
    body.copy_to_operands(assign);

    // i++ (V.3: synthetic size_type increment, built in IREP2)
    exprt incr =
      build_add(build_symbol(idx), gen_one(size_type()), size_type());
    code_assignt update(build_symbol(idx), incr);
    body.copy_to_operands(update);

    codet loop;
    loop.set_statement("while");
    loop.copy_to_operands(cond, body);
    converter_.add_instruction(loop);

    // Add null terminator at result[slice_len]
    exprt null_pos = build_index(build_symbol(result), slice_len, elem_type);
    code_assignt add_null(null_pos, gen_zero(elem_type));
    add_null.location() = location;
    converter_.add_instruction(add_null);

    return build_symbol(result);
  }

  // Handle list slicing
  symbolt &sliced_list = create_list();
  const locationt location = converter_.get_location_from_decl(list_value_);

  // Compute list size symbol once. Both bound resolution and (for negative
  // step) the size-1 default reuse this temporary instead of re-emitting
  // __ESBMC_list_size calls.
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  if (!size_func)
    throw std::runtime_error("__ESBMC_list_size not found in symbol table");

  exprt size_call = build_call_expr(
    *size_func,
    size_type(),
    {array.type().is_pointer() ? array : build_address_of(array)});
  size_call.location() = location;

  symbolt &size_sym = converter_.create_tmp_symbol(
    list_value_, "$slice_size$", size_type(), exprt());
  code_declt size_decl(build_symbol(size_sym));
  size_decl.copy_to_operands(size_call);
  converter_.add_instruction(size_decl);

  // Counter type: signed for negative step (must reach -1 sentinel without
  // unsigned underflow); unsigned otherwise to match the existing IR shape
  // for the common step==1 case.
  const typet signed_t = signed_size_type();
  const typet counter_type = negative_step ? signed_t : size_type();
  auto signed_add = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    plus_exprt out(lhs, rhs);
    out.type() = signed_t;
    return out;
  };
  auto signed_sub = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    minus_exprt out(lhs, rhs);
    out.type() = signed_t;
    return out;
  };

  // Resolve a slice bound following CPython's slice.indices(len) semantics.
  // Returns an expression of `counter_type`. We always work through signed
  // arithmetic so negative bounds (literal or runtime), the -1 stop sentinel,
  // and the clamp tree stay representable.
  auto resolve_bound = [&](const std::string &name, bool is_upper) -> exprt {
    const exprt size_signed = build_typecast(build_symbol(size_sym), signed_t);
    const exprt zero_s = from_integer(0, signed_t);
    const exprt one_s = from_integer(1, signed_t);
    const type2tc signed_t2 = migrate_type(signed_t); // V.3: for IREP2 selects

    // Step 1: produce a signed `resolved` value.
    //   - missing bound → step-direction default
    //   - present bound → val (signed); if val < 0 add size at runtime so the
    //     fix also covers non-literal negative bounds like xs[5:b:-1].
    exprt resolved;
    if (!slice_node.contains(name) || slice_node[name].is_null())
    {
      if (negative_step)
        resolved = is_upper ? from_integer(-1, signed_t)
                            : signed_sub(size_signed, one_s);
      else
        resolved = is_upper ? size_signed : zero_s;
    }
    else
    {
      exprt val_signed =
        build_typecast(converter_.get_expr(slice_node[name]), signed_t);
      exprt size_plus_val = signed_add(size_signed, val_signed);
      // val < 0 ? size + val : val  (V.3: built in IREP2).
      expr2tc val2, spv2;
      migrate_expr(val_signed, val2);
      migrate_expr(size_plus_val, spv2);
      resolved = migrate_expr_back(
        if2tc(signed_t2, lessthan2tc(val2, gen_zero(signed_t2)), spv2, val2));
    }

    // Step 2: clamp to the CPython-defined window for this step direction.
    //   pos step: [0, size]                  (start and stop)
    //   neg step start: [-1, size-1]
    //   neg step stop:  [-1, size-1] (stop>=size is harmless — the loop
    //                                terminates immediately — but we clamp
    //                                to keep the expression in a known range)
    const exprt under = negative_step ? from_integer(-1, signed_t) : zero_s;
    const exprt over =
      negative_step ? signed_sub(size_signed, one_s) : size_signed;

    // c1 = max(resolved, under); c2 = min(c1, over)  (V.3: built in IREP2).
    expr2tc resolved2, under2, over2;
    migrate_expr(resolved, resolved2);
    migrate_expr(under, under2);
    migrate_expr(over, over2);
    expr2tc c1 =
      if2tc(signed_t2, lessthan2tc(resolved2, under2), under2, resolved2);
    expr2tc c2 = if2tc(signed_t2, greaterthan2tc(c1, over2), over2, c1);
    exprt c2_legacy = migrate_expr_back(c2);

    return negative_step ? c2_legacy : build_typecast(c2_legacy, size_type());
  };

  exprt lower_expr = resolve_bound("lower", false);
  exprt upper_expr = resolve_bound("upper", true);

  // Initialize counter at lower.
  symbolt &counter = converter_.create_tmp_symbol(
    list_value_, "counter", counter_type, lower_expr);
  code_assignt counter_init(build_symbol(counter), lower_expr);
  converter_.add_instruction(counter_init);

  // Loop condition: counter < upper (positive step) or counter > upper
  // (negative step). Negative step uses signed comparison thanks to the
  // signed counter/upper type.
  exprt loop_condition(negative_step ? ">" : "<", bool_type());
  loop_condition.copy_to_operands(build_symbol(counter), upper_expr);

  // Loop body
  code_blockt loop_body;

  exprt index_expr = build_symbol(counter);
  if (negative_step)
    index_expr = build_typecast(index_expr, size_type());

  const exprt list_at_call = build_list_at_call(array, index_expr, list_value_);
  const symbolt &at_result = converter_.create_tmp_symbol(
    list_value_,
    "tmp_list_at",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());

  code_declt at_decl(build_symbol(at_result));
  at_decl.copy_to_operands(list_at_call);
  loop_body.copy_to_operands(at_decl);

  // Shallow append: preserve element value pointers so nested lists survive the
  // slice copy uncorrupted (esbmc/esbmc#5102).
  const symbolt *push_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
  if (!push_func)
    throw std::runtime_error("Push function symbol not found");

  // Nested-list elements keep their inner pointer; scalars are byte-copied, so
  // the slice copy does not corrupt nested lists (#5102).
  constant_exprt slice_list_type_id(size_type());
  slice_list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
      converter_.get_type_handler().get_list_type())),
    config.ansi_c.address_width));
  exprt push_call = build_call_expr(
    *push_func,
    bool_type(),
    {build_symbol(sliced_list), build_symbol(at_result), slice_list_type_id});
  push_call.location() = location;
  loop_body.copy_to_operands(converter_.convert_expression_to_code(push_call));

  // counter += step_val. step_val is signed; for positive step it's >0 and
  // converts to size_type cleanly via from_integer.
  exprt step_const = from_integer(step_val, counter_type);
  exprt increment = build_add(build_symbol(counter), step_const, counter_type);
  code_assignt counter_update(build_symbol(counter), increment);
  loop_body.copy_to_operands(counter_update);

  codet while_loop;
  while_loop.set_statement("while");
  while_loop.copy_to_operands(loop_condition, loop_body);
  converter_.add_instruction(while_loop);

  // Update type map for sliced elements. Element types are taken positionally
  // from the source list literal. This is valid when step == 1 and each bound
  // is either absent (defaulting to the full range) or a non-negative constant
  // literal. Absent bounds cover the idiomatic full copy src[:] and half-open
  // slices like src[1:] / src[:2], whose nested-list element types would
  // otherwise be lost — corrupting reads such as sl[0][0] (esbmc/esbmc#5102).
  // For step != 1 the index mapping is no longer lower + i, and for negative or
  // non-constant bounds the static range is unknown; both keep the generic
  // fallback below.
  bool bounds_usable = true;
  bool has_lower = false;
  bool has_upper = false;
  size_t lower_bound = 0;
  size_t upper_bound = 0;
  for (int which = 0; which < 2; ++which)
  {
    const char *key = which == 0 ? "lower" : "upper";
    bool &present = which == 0 ? has_lower : has_upper;
    size_t &out = which == 0 ? lower_bound : upper_bound;

    present = slice_node.contains(key) && !slice_node[key].is_null();
    if (!present)
      continue; // absent / None: defaults to the list bound below
    const nlohmann::json &b = slice_node[key];
    if (
      b.is_object() && b.contains("_type") && b["_type"] == "Constant" &&
      b.contains("value") && b["value"].is_number_integer() &&
      b["value"].get<long long>() >= 0)
      out = b["value"].get<size_t>();
    else
      bounds_usable = false; // negative or non-constant: not known statically
  }

  if (step_val == 1 && bounds_usable)
  {
    nlohmann::json list_node;
    if (
      list_value_.contains("value") && list_value_["value"].is_object() &&
      list_value_["value"].contains("id") &&
      list_value_["value"]["id"].is_string())
    {
      list_node = json_utils::get_var_value(
        list_value_["value"]["id"],
        converter_.current_function_name(),
        converter_.ast());
    }

    // Only update type map for actual lists (not strings or other types)
    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array())
    {
      const size_t elts_size = list_node["value"]["elts"].size();
      const size_t begin = has_lower ? std::min(lower_bound, elts_size) : 0;
      const size_t end =
        has_upper ? std::min(upper_bound, elts_size) : elts_size;

      for (size_t i = begin; i < end; ++i)
      {
        exprt element = converter_.get_expr(list_node["value"]["elts"][i]);
        if (!element.is_symbol())
        {
          symbolt &tmp = converter_.create_tmp_symbol(
            list_value_, "$slice_elem$", element.type(), element);
          code_declt decl(build_symbol(tmp));
          decl.location() = location;
          converter_.add_instruction(decl);

          code_assignt assign(build_symbol(tmp), element);
          assign.location() = location;
          converter_.add_instruction(assign);
          element = build_symbol(tmp);
        }

        list_type_map[sliced_list.id.as_string()].push_back(
          std::make_pair(element.identifier().as_string(), element.type()));
      }
    }
  }

  // This handles cases where one or both bounds are null or negative (e.g.
  // numbers[:-1]), or where the source is a function parameter rather than a
  // locally constructed list, so list_type_map has no entries for it.
  const std::string &sliced_id = sliced_list.id.as_string();
  if (list_type_map[sliced_id].empty())
  {
    if (
      list_type_map[sliced_id].empty() && list_value_.contains("value") &&
      list_value_["value"].contains("id"))
    {
      const std::string &param_name =
        list_value_["value"]["id"].get<std::string>();

      nlohmann::json param_node = json_utils::find_var_decl(
        param_name, converter_.current_function_name(), converter_.ast());

      // Only use annotation fallback for function parameters (arg nodes),
      // not for local variable declarations whose element type should come
      // from the type map populated during list construction.
      if (!param_node.is_null() && param_node["_type"] == "arg")
      {
        const typet elem_type = get_elem_type_from_annotation(
          param_node, converter_.get_type_handler());

        if (elem_type != typet())
        {
          list_type_map[sliced_id].push_back(
            std::make_pair(std::string(), elem_type));
        }
      }
    }
  }

  return build_symbol(sliced_list);
}

void python_list::handle_slice_assignment(
  const exprt &list_expr,
  const nlohmann::json &slice_node,
  const nlohmann::json &value_node)
{
  // The step is restricted to a constant integer literal (or absent) for
  // now; the model receives it as a runtime value and branches on it at
  // solve time, so this could be lifted, but a symbolic step is untested.
  // Mirrors the literal-only step extraction in handle_range_slice, but
  // rejects a non-constant step instead of silently slicing with step 1.
  long long step_val = 1;
  if (slice_node.contains("step") && !slice_node["step"].is_null())
  {
    const auto &step_node = slice_node["step"];
    if (
      step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub" &&
      step_node["operand"]["_type"] == "Constant" &&
      step_node["operand"]["value"].is_number_integer())
      step_val = -(long long)step_node["operand"]["value"].get<std::int64_t>();
    else if (
      step_node["_type"] == "Constant" &&
      step_node["value"].is_number_integer())
      step_val = step_node["value"].get<std::int64_t>();
    else
      throw std::runtime_error(
        "List slice assignment requires a constant integer step");
  }

  const locationt location = converter_.get_location_from_decl(slice_node);

  // Evaluate the right-hand side; materialize a call (e.g. sorted(...)) into
  // a temporary so its result can be passed to the model by address. get_expr
  // returns calls as code_function_callt, which must be emitted as an
  // instruction with the temporary as its lhs — embedding it as a decl
  // operand would leak a side effect into the GOTO program.
  exprt rhs = converter_.get_expr(value_node);
  if (rhs.is_function_call())
  {
    code_function_callt &fcall = static_cast<code_function_callt &>(rhs);
    if (!fcall.function().type().is_code())
      throw std::runtime_error(
        "Unsupported callable on list slice assignment right-hand side");
    const typet ret_type = to_code_type(fcall.function().type()).return_type();
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", ret_type, exprt());
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);
    fcall.lhs() = build_symbol(tmp);
    converter_.add_instruction(fcall);
    rhs = build_symbol(tmp);
  }
  else if (rhs.id() == "sideeffect")
  {
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", rhs.type(), exprt());
    code_declt decl(build_symbol(tmp));
    decl.copy_to_operands(rhs);
    decl.location() = location;
    converter_.add_instruction(decl);
    rhs = build_symbol(tmp);
  }

  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_list_type = ns.follow(list_type);
  const typet rhs_type = ns.follow(rhs.type());
  const bool rhs_is_list =
    rhs_type == resolved_list_type ||
    (rhs_type.is_pointer() &&
     ns.follow(rhs_type.subtype()) == resolved_list_type);
  if (!rhs_is_list)
    throw std::runtime_error(
      "List slice assignment requires a list right-hand side");

  const typet i64 = signedbv_typet(64);
  auto bound_expr = [&](const char *name, bool &present) -> exprt {
    present = slice_node.contains(name) && !slice_node[name].is_null();
    if (!present)
      return from_integer(0, i64);
    exprt e = converter_.get_expr(slice_node[name]);
    e = remove_function_calls_recursive(e, slice_node);
    return build_typecast(e, i64);
  };
  bool has_lower = false, has_upper = false;
  exprt lower = bound_expr("lower", has_lower);
  exprt upper = bound_expr("upper", has_upper);

  const symbolt *fn =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_slice_assign");
  if (!fn)
    throw std::runtime_error(
      "__ESBMC_list_slice_assign not found in symbol table");

  // Same nested-list type id as the slice-read path: lets the model keep
  // nested-list records shared instead of byte-copying them (#5102).
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(list_type)),
    config.ansi_c.address_width));

  // Statically-known element byte size, passed to the model so its per-element
  // copy uses a compile-time-constant size and takes __ESBMC_copy_value's
  // branch-free fast path, instead of reading the symbolic-index field
  // src->items[k].size (which forces memcpy's per-byte loop to unwind per
  // element — the dominant cost on large slice assignments). Emitted ONLY for
  // an unambiguous fixed-width scalar element type; 0 otherwise, which makes
  // the model fall back to the per-element o->size read (exact prior behaviour,
  // so a missing/ambiguous type can never produce a wrong copy length).
  // Derive the element byte size from the RHS source list's elements so the
  // model copies with a compile-time-constant size (fast path) instead of the
  // symbolic-index field read src->items[k].size. We read the *converted* RHS
  // element type, which is the actual data being stored, so the size is always
  // correct. Only a plain fixed-width scalar (bitvector) is emitted; anything
  // else (pointers, structs, strings, nested lists, non-literal/heterogeneous
  // RHS) keeps elem_size 0, making the model fall back to the per-element
  // o->size read — exact prior behaviour, so a wrong length is impossible.
  auto scalar_width = [](const typet &t) -> size_t {
    if (
      (t.id() == "signedbv" || t.id() == "unsignedbv" || t.id() == "floatbv" ||
       t.id() == "fixedbv") &&
      !t.width().empty())
      return std::stoull(t.width().as_string(), nullptr, 10) / 8;
    return 0;
  };
  size_t elem_size_bytes = 0;
  if (
    value_node.contains("_type") && value_node["_type"] == "List" &&
    value_node.contains("elts") && value_node["elts"].is_array() &&
    !value_node["elts"].empty())
  {
    // RHS is a list literal: size from its (uniform) element types.
    bool uniform = true;
    size_t w = 0;
    for (const auto &elt : value_node["elts"])
    {
      size_t ew = 0;
      try
      {
        ew = scalar_width(converter_.get_expr(elt).type());
      }
      catch (...)
      {
        uniform = false;
        break;
      }
      if (ew == 0 || (w != 0 && ew != w))
      {
        uniform = false;
        break;
      }
      w = ew;
    }
    if (uniform)
      elem_size_bytes = w;
  }
  else if (
    value_node.contains("_type") && value_node["_type"] == "Name" &&
    value_node.contains("id"))
  {
    // RHS is a list variable (includes self-assignment l[...] = l): size from
    // its declared element type.
    const typet et = get_list_element_type(value_node["id"].get<std::string>());
    if (!et.is_nil())
      elem_size_bytes = scalar_width(et);
  }
  constant_exprt elem_size(size_type());
  elem_size.set_value(
    integer2binary(BigInt(elem_size_bytes), config.ansi_c.address_width));

  exprt call = build_call_expr(
    *fn,
    bool_type(),
    {list_expr.type().is_pointer() ? list_expr : build_address_of(list_expr),
     lower,
     from_integer(has_lower ? 1 : 0, int_type()),
     upper,
     from_integer(has_upper ? 1 : 0, int_type()),
     from_integer(step_val, i64),
     rhs.type().is_pointer() ? rhs : build_address_of(rhs),
     list_type_id,
     elem_size});
  call.location() = location;
  converter_.add_instruction(converter_.convert_expression_to_code(call));
}

exprt python_list::handle_index_access(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());

  // Find list node for type information
  nlohmann::json list_node;
  if (
    list_value_.contains("value") && list_value_["value"].is_object() &&
    list_value_["value"].contains("id"))
  {
    list_node = json_utils::find_var_decl(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());
  }

  exprt pos_expr = converter_.get_expr(slice_node);
  pos_expr = converter_.unwrap_optional_if_needed(pos_expr);
  size_t index = 0;

  // Validate index type
  if (pos_expr.type().is_array())
  {
    locationt l = converter_.get_location_from_decl(list_value_);
    throw std::runtime_error(
      "TypeError at " + l.get_file().as_string() + " " +
      l.get_line().as_string() +
      ": list indices must be integers or slices, not str");
  }

  // Handle negative indices
  if (slice_node.contains("op") && slice_node["op"]["_type"] == "USub")
  {
    // Both compile-time branches below assume the negated operand is a
    // constant literal (a[-1]). For a non-constant operand (a[-i]) the value
    // is only known at runtime, so leave pos_expr (= -i) and index untouched:
    // build_list_at_call normalizes the negative index at runtime via
    // __ESBMC_list_size, and the element-type lookup falls back to element 0,
    // which is correct for the homogeneous lists ESBMC models (#4926).
    const bool operand_is_constant =
      slice_node.contains("operand") &&
      slice_node["operand"]["_type"] == "Constant" &&
      slice_node["operand"].contains("value");

    if (!operand_is_constant)
    {
      // Nothing to do: runtime normalization handles a[-i].
    }
    // For char* (string parameters), skip compile-time normalization: the size
    // is not known statically, so normalization happens at runtime in the
    // char* indexing block below.
    else if (
      !array.type().is_pointer() &&
      (list_node.is_null() || list_node["value"]["_type"] != "List"))
    {
      BigInt v = binary2integer(pos_expr.op0().value().c_str(), true);
      v *= -1;

      const array_typet &t = static_cast<const array_typet &>(array.type());
      BigInt s = binary2integer(t.size().value().c_str(), true);

      // For char arrays (strings), exclude null terminator from logical length
      if (t.subtype() == char_type())
        s -= 1;

      v += s;
      pos_expr = from_integer(v, pos_expr.type());
    }
    else
    {
      // Compute index for compile-time type lookup only.
      // Do NOT overwrite pos_expr: the list may have been mutated
      // (append/insert/extend), so we must resolve the negative index
      // at runtime via build_list_at_call using __ESBMC_list_size.
      index = slice_node["operand"]["value"].get<size_t>();
      index = list_node["value"]["elts"].size() - index;
    }
  }
  else if (slice_node["_type"] == "Constant")
  {
    index = slice_node["value"].get<size_t>();
  }

  // Handle different array types
  const bool is_char_array = resolved_array_type.is_array() &&
                             resolved_array_type.subtype() == char_type();
  const bool is_char_ptr = resolved_array_type.is_pointer() &&
                           resolved_array_type.subtype() == char_type();

  if (
    (array.type().is_symbol() || array.type().subtype().is_symbol()) &&
    !is_char_array && !is_char_ptr)
  {
    // Handle list types (symbol-based)
    typet elem_type;

    // Check for nested list access
    if (array.type() == converter_.get_type_handler().get_list_type())
    {
      const auto &key = array.identifier().as_string();
      auto type_map_it = list_type_map.find(key);
      if (type_map_it != list_type_map.end())
      {
        if (!type_map_it->second.empty())
        {
          // Homogeneous lists (e.g. list comprehensions) record a single
          // element-type entry; ESBMC models lists as homogeneous, so reuse
          // that entry for any in-structure index.  Without this, a constant
          // outer index >= 1 into a comprehension-built nested list skips the
          // nested-element handling below, the inner element type (float) is
          // lost, and the value reaches the SMT FP encoder as a non-FP sort
          // (get_exponent_width abort / Z3 "rm and fp sorts", #5129).  Literal
          // lists record one entry per element, so for an in-bounds index
          // eff_index == index and behaviour is unchanged.  The runtime value
          // is still read at pos_expr below, so only the static type is taken
          // from the homogeneous entry.
          const size_t eff_index = index < type_map_it->second.size()
                                     ? index
                                     : type_map_it->second.size() - 1;
          const std::string &elem_id = type_map_it->second.at(eff_index).first;
          elem_type = type_map_it->second.at(eff_index).second;

          if (elem_type == converter_.get_type_handler().get_list_type())
          {
            // Nested-list element.  The recorded elem_id names the inner-list
            // symbol, but copy_type_info copies that id verbatim across a
            // function-return boundary (Q = build()), where it is a *callee*
            // frame local — returning build_symbol(elem_id) would reference a
            // symbol that is never assigned in the caller's symex frame, so its
            // value is nondet (float_buf OOB / wrong value, #5103/#5102).
            //
            // Instead, read the inner list pointer at runtime from `array`
            // (valid in every scope: list_at returns the stored pointer, so
            // aliasing/mutation semantics are preserved), bind it to a fresh
            // caller-scope symbol, and copy the inner element-type map onto
            // that symbol so deeper subscripts (Q[i][j]) still resolve
            // int/float.  For parameter-annotation-only entries (elem_id
            // empty) fall through to the dynamic __ESBMC_list_at path below.
            if (!elem_id.empty())
            {
              const locationt loc =
                converter_.get_location_from_decl(list_value_);

              exprt list_at_call =
                build_list_at_call(array, pos_expr, list_value_);
              exprt inner_ptr = extract_pyobject_value(list_at_call, elem_type);

              symbolt &inner_sym = converter_.create_tmp_symbol(
                list_value_, "$nested_list$", elem_type, exprt());
              code_declt inner_decl(build_symbol(inner_sym));
              inner_decl.location() = loc;
              converter_.add_instruction(inner_decl);

              code_assignt inner_assign(build_symbol(inner_sym), inner_ptr);
              inner_assign.location() = loc;
              converter_.add_instruction(inner_assign);

              copy_type_info(elem_id, inner_sym.id.as_string());
              return build_symbol(inner_sym);
            }
          }
        }
      }
    }

    // Determine element type
    if (list_node.contains("_type") && list_node["_type"] == "arg")
    {
      elem_type =
        get_elem_type_from_annotation(list_node, converter_.get_type_handler());
    }
    else if (
      slice_node["_type"] == "Constant" || slice_node["_type"] == "BinOp" ||
      (slice_node["_type"] == "UnaryOp" &&
       slice_node["operand"]["_type"] == "Constant"))
    {
      const std::string &list_name = array.identifier().as_string();

      if (list_type_map[list_name].empty())
      {
        /* Fall back to annotation for function parameters */
        if (
          list_value_.contains("value") && list_value_["value"].is_object() &&
          list_value_["value"].contains("id") &&
          list_value_["value"]["id"].is_string())
        {
          const nlohmann::json list_value_node = json_utils::get_var_value(
            list_value_["value"]["id"],
            converter_.current_function_name(),
            converter_.ast());

          elem_type = get_elem_type_from_annotation(
            list_value_node, converter_.get_type_handler());
        }
      }
      else
      {
        size_t type_index =
          (!list_node.is_null() && list_node.contains("value") &&
           list_node["value"].is_object() &&
           list_node["value"].contains("_type") &&
           list_node["value"]["_type"] == "BinOp")
            ? 0
            : index;

        try
        {
          elem_type = list_type_map[list_name].at(type_index).second;
        }
        catch (const std::out_of_range &)
        {
          // Do not emit a frontend conversion error for constant OOB indices.
          // The runtime list access model can raise IndexError, which is
          // observable by Python try/except code.

          // Use the known element type for homogeneous dynamic lists.
          if (!list_type_map[list_name].empty())
          {
            elem_type = list_type_map[list_name].back().second;
          }
          else if (
            list_value_.contains("value") &&
            list_value_["value"].contains("id") &&
            list_value_["value"]["id"].is_string())
          {
            // Try annotation fallback for dynamic lists or function parameters
            const nlohmann::json list_value_node = json_utils::get_var_value(
              list_value_["value"]["id"],
              converter_.current_function_name(),
              converter_.ast());

            elem_type = get_elem_type_from_annotation(
              list_value_node, converter_.get_type_handler());
          }

          // Only throw if annotation also fails
          if (elem_type == typet())
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }
    }
    else if (slice_node["_type"] == "Name")
    {
      // First try to get element type from list_node if it's an AnnAssign
      if (
        !list_node.is_null() && list_node["_type"] == "AnnAssign" &&
        list_node.contains("annotation") && elem_type == typet())
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }

      // If still no elem_type, try to get it from the array variable's type annotation
      if (array.is_symbol() && elem_type == typet())
      {
        // Extract variable name from the symbol identifier
        std::string list_var_name = json_utils::extract_var_name_from_symbol_id(
          array.identifier().as_string());

        // Find the variable's declaration to check for type annotation
        nlohmann::json list_var_decl = json_utils::find_var_decl(
          list_var_name, converter_.current_function_name(), converter_.ast());

        // If the variable has a type annotation such as list[str], extract element type
        if (!list_var_decl.is_null() && list_var_decl.contains("annotation"))
        {
          elem_type = get_elem_type_from_annotation(
            list_var_decl, converter_.get_type_handler());
        }
      }

      // Handle variable-based indexing
      if (
        elem_type == typet() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("_type") &&
        list_node["value"]["_type"] == "Call")
      {
        elem_type =
          infer_elem_type_from_call_return(list_node["value"], converter_);
        if (elem_type != typet() && array.is_symbol())
        {
          const std::string &list_name = array.identifier().as_string();
          if (list_type_map[list_name].empty())
            list_type_map[list_name].push_back(std::make_pair("", elem_type));
        }
      }

      if (!list_node.is_null() && list_node["_type"] == "arg")
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }
      else
      {
        // Handle case where we need to find the variable declaration
        while (!list_node.is_null() && (!list_node.contains("value") ||
                                        !list_node["value"].contains("elts") ||
                                        !list_node["value"]["elts"].is_array()))
        {
          if (list_node.contains("value") && list_node["value"].contains("id"))
            list_node = json_utils::find_var_decl(
              list_node["value"]["id"],
              converter_.current_function_name(),
              converter_.ast());
          else
          {
            break;
          }
        }

        if (!list_node.is_null() && list_node["_type"] == "arg")
        {
          elem_type = get_elem_type_from_annotation(
            list_node, converter_.get_type_handler());
        }
        else if (!list_node.is_null() && list_node.contains("value"))
        {
          // Check if the value is a Subscript (such as d['a'])
          if (
            list_node["value"].contains("_type") &&
            list_node["value"]["_type"] == "Subscript")
          {
            // For ESBMC_iter_0 = d['a'], get element type from dict's actual value
            if (
              list_node["value"].contains("value") &&
              list_node["value"]["value"].is_object() &&
              list_node["value"]["value"].contains("_type") &&
              list_node["value"]["value"]["_type"] == "Name")
            {
              std::string dict_var_name =
                list_node["value"]["value"]["id"].get<std::string>();

              // Find the dict's declaration
              nlohmann::json dict_node = json_utils::find_var_decl(
                dict_var_name,
                converter_.current_function_name(),
                converter_.ast());

              if (!dict_node.is_null() && dict_node.contains("value"))
              {
                const auto &dict_value = dict_node["value"];

                // Get the key being accessed (e.g., 'a' in d['a'])
                if (list_node["value"].contains("slice"))
                {
                  const auto &key_node = list_node["value"]["slice"];

                  // Handle constant string key
                  if (
                    key_node["_type"] == "Constant" &&
                    key_node.contains("value"))
                  {
                    std::string key = key_node["value"].get<std::string>();

                    // For dict literals, get the corresponding value
                    if (
                      dict_value["_type"] == "Dict" &&
                      dict_value.contains("keys") &&
                      dict_value.contains("values"))
                    {
                      const auto &keys = dict_value["keys"];
                      const auto &values = dict_value["values"];

                      // Find the matching key
                      for (size_t i = 0; i < keys.size(); i++)
                      {
                        if (
                          keys[i]["_type"] == "Constant" &&
                          keys[i]["value"].get<std::string>() == key)
                        {
                          // Found the value: now get its element type
                          // (promotion-aware, see infer_literal_element_type).
                          elem_type = infer_literal_element_type(values[i]);
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          else if (
            list_node["value"].contains("elts") &&
            list_node["value"]["elts"].is_array() &&
            !list_node["value"]["elts"].empty())
          {
            // Infer element type from the literal, accounting for the int->float
            // promotion applied to mixed numeric literals at construction.
            elem_type = infer_literal_element_type(list_node["value"]);
          }
        }
      }
    }

    // For variable indices, prefer compile-time list element type information
    // when available.  Also fire when elem_type is empty_typet(): that is the
    // sentinel returned by get_typet("tuple"), meaning "don't use the
    // annotation as-is; resolve the concrete struct type via list_type_map."
    if ((elem_type == typet() || elem_type.is_empty()) && array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      auto type_map_it = list_type_map.find(list_name);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
        elem_type = type_map_it->second.back().second;
    }

    // Nested subscript chains like W[i][j] where the base is a Name with a
    // nested list annotation (e.g. list[list[T]]).  list_node is null in this
    // case because list_value_["value"] is itself a Subscript rather than a
    // Name, so the standard paths above can't see W's annotation.  Walk the
    // Subscript chain down to the base Name and peel that many layers off the
    // annotation, then hand the result to the existing extractor.
    if (elem_type == typet())
    {
      size_t subscript_depth = 0;
      const nlohmann::json *cur = &list_value_;
      while (cur->is_object() && cur->contains("value") &&
             (*cur)["value"].is_object() && (*cur)["value"].contains("_type") &&
             (*cur)["value"]["_type"] == "Subscript")
      {
        ++subscript_depth;
        cur = &(*cur)["value"];
      }
      if (
        subscript_depth > 0 && cur->is_object() && cur->contains("value") &&
        (*cur)["value"].is_object() && (*cur)["value"].contains("id") &&
        (*cur)["value"]["id"].is_string())
      {
        nlohmann::json base_decl = json_utils::find_var_decl(
          (*cur)["value"]["id"].get<std::string>(),
          converter_.current_function_name(),
          converter_.ast());
        if (!base_decl.is_null() && base_decl.contains("annotation"))
        {
          nlohmann::json drilled = base_decl["annotation"];
          for (size_t k = 0; k < subscript_depth; ++k)
          {
            if (
              !drilled.is_object() || !drilled.contains("_type") ||
              drilled["_type"] != "Subscript" || !drilled.contains("slice"))
            {
              drilled = nlohmann::json();
              break;
            }
            drilled = drilled["slice"];
          }
          if (!drilled.is_null())
          {
            nlohmann::json synth;
            synth["annotation"] = drilled;
            elem_type = get_elem_type_from_annotation(
              synth, converter_.get_type_handler());
          }
        }
      }
    }

    // Python allows indexing lists with unknown element type in dynamic code.
    // If we resolved the index but not the element type, treat elements as Any
    // instead of raising a frontend conversion error.
    if (pos_expr != exprt() && elem_type == typet())
      elem_type = any_type();

    if (pos_expr == exprt() || elem_type == typet())
    {
      const bool has_const_index =
        (slice_node["_type"] == "Constant" && slice_node.contains("value")) ||
        (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
         slice_node["op"]["_type"] == "USub" &&
         slice_node.contains("operand") &&
         slice_node["operand"]["_type"] == "Constant");
      if (has_const_index)
      {
        // If we can prove out-of-bounds from a concrete list literal, raise
        // IndexError in-model so Python try/except(IndexError) can catch it.
        if (
          array.is_symbol() && !list_node.is_null() &&
          list_node.contains("value") && list_node["value"].is_object() &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          bool negative_index = false;
          size_t index_abs = 0;

          if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
          {
            index_abs = slice_node["value"].get<size_t>();
          }
          else
          {
            negative_index = true;
            index_abs = slice_node["operand"]["value"].get<size_t>();
          }

          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);

          if (out_of_bounds)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Unreachable after raise, but return a placeholder expression.
            return gen_zero(size_type());
          }
        }

        // If array is a constant placeholder (e.g., from a chained OOB access),
        // we're in dead code after a prior IndexError. Emit IndexError and return
        // a placeholder rather than crashing the frontend.
        if (!array.is_symbol() && array.is_constant())
        {
          exprt raise = converter_.get_exception_handler().gen_exception_raise(
            "IndexError", "list index out of range");
          codet throw_code("expression");
          throw_code.operands().push_back(raise);
          converter_.add_instruction(throw_code);
          return gen_zero(size_type());
        }

        const locationt l = converter_.get_location_from_decl(list_value_);
        throw std::runtime_error(
          "List out of bounds at " + l.get_file().as_string() +
          " line: " + l.get_line().as_string());
      }

      // Keep historical frontend diagnostic for literal lists with
      // compile-time constant OOB indexing (including nested accesses).
      if (
        array.is_symbol() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("elts") &&
        list_node["value"]["elts"].is_array())
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        if (has_const_index)
        {
          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);
          if (out_of_bounds)
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }

      throw std::runtime_error(
        "Invalid list access: could not resolve position or element type");
    }

    // Preserve historical frontend OOB diagnostics only when we can prove we
    // are indexing a stable list literal (not a reassigned/mutated/derived
    // list). Using the type map alone is too broad and causes false positives.
    // Emit an IndexError raise instead of a C++ exception so Python
    // try/except(IndexError) can observe the error.
    if (array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      auto it = list_type_map.find(list_name);
      if (it != list_type_map.end() && !it->second.empty())
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        const bool is_slice_derived_var =
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "Subscript";

        bool is_stable_list_literal = false;
        if (
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "List" &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          // If sizes diverge, the symbol was likely reassigned/mutated.
          is_stable_list_literal =
            it->second.size() == list_node["value"]["elts"].size();
        }

        if (has_const_index && !is_slice_derived_var && is_stable_list_literal)
        {
          const size_t known_size = it->second.size();
          const bool oob = negative_index ? (index_abs > known_size)
                                          : (index_abs >= known_size);
          if (oob)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Short-circuit: the raise makes the rest unreachable.
            return gen_zero(elem_type);
          }
        }
      }
    }

    // A non-constant subscript into a heterogeneous int/float list cannot
    // resolve a single static element type from the index, so the code above
    // defaults to element 0's type. That misreads any element of the other
    // numeric kind (e.g. a float stored at index 2 read as an int, #5160).
    // Python promotes int to float in numeric expressions, so treat the read
    // as float and dispatch on the stored type_id at runtime (see
    // extract_pyobject_value), which yields the correct value for both kinds.
    const bool constant_index =
      slice_node["_type"] == "Constant" ||
      (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
       slice_node["op"]["_type"] == "USub" && slice_node.contains("operand") &&
       slice_node["operand"]["_type"] == "Constant");
    const bool mixed_numeric =
      !constant_index && array.is_symbol() &&
      has_mixed_numeric_types(array.identifier().as_string());
    if (mixed_numeric)
      elem_type = double_type();

    // A heap-migrated user-class instance is stored in a list as a `Class*`
    // pointer: the push path records a pointer-sized element (type_size 8), not
    // the full struct. If the element type resolved to a by-value user-class
    // struct, read it back as the pointer it actually is so
    // extract_pyobject_value dereferences a single `Class*` instead of copying
    // sizeof(struct) bytes off an 8-byte pointer slot, which overruns it
    // (#4805). ESBMC-internal model helper classes (reserved `__ESBMC_` prefix,
    // e.g. the dataclasses `__ESBMC_DataclassField`) are stored by value by
    // their hand-written models and must be left as structs.
    if (converter_.is_user_class_struct_type(elem_type))
    {
      const std::string tag =
        elem_type.id() == "symbol"
          ? to_symbol_type(elem_type).get_identifier().as_string()
          : to_struct_type(elem_type).tag().as_string();
      const std::string cls = converter_.extract_class_name_from_tag(tag);
      if (cls.rfind("__ESBMC", 0) != 0)
        elem_type = gen_pointer_type(elem_type);
    }

    // Build list access and cast result
    exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);

    // The mixed-numeric read dereferences the element three times (type_id,
    // float_idx, value), so bind the __ESBMC_list_at result to a temp first and
    // evaluate the access once instead of three times on the hot path.
    if (mixed_numeric)
    {
      const locationt loc = converter_.get_location_from_decl(list_value_);
      symbolt &elem_obj = converter_.create_tmp_symbol(
        list_value_,
        "$list_elem_obj$",
        pointer_typet(converter_.get_type_handler().get_list_element_type()),
        exprt());
      code_declt obj_decl(build_symbol(elem_obj));
      obj_decl.location() = loc;
      converter_.add_instruction(obj_decl);
      code_assignt obj_assign(build_symbol(elem_obj), list_at_call);
      obj_assign.location() = loc;
      converter_.add_instruction(obj_assign);
      list_at_call = build_symbol(elem_obj);
    }

    // Extract and dereference PyObject value. string_safe: a subscript read is
    // value-context (assigned to the target), so an any_type string element may
    // be returned by pointer without overrunning its short char[] storage.
    return extract_pyobject_value(
      list_at_call, elem_type, mixed_numeric, /*string_safe=*/true);
  }

  // Handle static string indexing with IndexError on out-of-bounds
  if (is_char_array)
  {
    exprt idx = pos_expr;
    if (idx.type() != size_type())
      idx = build_typecast(idx, size_type());

    // Logical string length excludes the null terminator
    exprt array_size = to_array_type(resolved_array_type).size();
    if (array_size.type() != size_type())
      array_size = build_typecast(array_size, size_type());
    // str_len = array_size - 1. array_size is typecast to size_type above, so
    // both operands share width; build the subtraction in IREP2 (V.3).
    exprt str_len =
      build_sub(array_size, from_integer(1, size_type()), size_type());

    // Emit: if (idx >= str_len) throw IndexError("string index out of range")
    // (idx is typecast to size_type above; str_len is a size_type subtraction —
    // both operands share width.)
    exprt oob_cond = build_greater_equal(idx, str_len);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = oob_cond;
    guard.then_case() = throw_code;
    converter_.add_instruction(guard);

    // Tag with #cpp_type==char so downstream consumers (notably
    // python_converter::get_python_type_category) can distinguish a 1-char
    // string element from an arbitrary 8-bit int (e.g. dtype=np.int8) without
    // resorting to a fragile width-only heuristic.
    typet char_t = char_type();
    type_utils::set_cpp_type(char_t, "char");
    return build_index(array, idx, char_t);
  }

  // For char* (string function parameter), implement Python single-index
  // semantics: normalize negative indices, raise IndexError on out-of-bounds,
  // then return a fresh single-char null-terminated string via
  // __python_str_slice so the result is never a dangling pointer.
  if (is_char_ptr)
  {
    typet ll_type = signedbv_typet(64);
    locationt loc = converter_.get_location_from_decl(list_value_);

    // --- 1. strlen(array) → len_sym ---
    const symbolt *strlen_sym =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (!strlen_sym)
      throw std::runtime_error("strlen not found in symbol table");

    symbolt &len_sym = converter_.create_tmp_symbol(
      list_value_, "$str_len$", size_type(), gen_zero(size_type()));
    code_declt len_decl(build_symbol(len_sym));
    len_decl.location() = loc;
    converter_.add_instruction(len_decl);

    code_function_callt strlen_call;
    strlen_call.function() = build_symbol(*strlen_sym);
    strlen_call.lhs() = build_symbol(len_sym);
    strlen_call.arguments().push_back(array);
    strlen_call.type() = size_type();
    strlen_call.location() = loc;
    converter_.add_instruction(strlen_call);

    // --- 2. idx_sym = (long long)pos_expr ---
    symbolt &idx_sym = converter_.create_tmp_symbol(
      list_value_, "$str_idx$", ll_type, gen_zero(ll_type));
    code_declt idx_decl(build_symbol(idx_sym));
    idx_decl.location() = loc;
    converter_.add_instruction(idx_decl);

    code_assignt idx_init(
      build_symbol(idx_sym), build_typecast(pos_expr, ll_type));
    idx_init.location() = loc;
    converter_.add_instruction(idx_init);

    // --- 3. Normalize negative index: if (idx < 0) idx += (ll)len ---
    // ($str_idx$ and the 0 literal are both ll_type — same width)
    exprt idx_lt_zero =
      build_less_than(build_symbol(idx_sym), from_integer(0, ll_type));

    exprt idx_plus_len = build_add(
      build_symbol(idx_sym),
      build_typecast(build_symbol(len_sym), ll_type),
      ll_type);

    code_assignt normalize(build_symbol(idx_sym), idx_plus_len);
    normalize.location() = loc;

    code_ifthenelset norm_guard;
    norm_guard.cond() = idx_lt_zero;
    norm_guard.then_case() = normalize;
    norm_guard.location() = loc;
    converter_.add_instruction(norm_guard);

    // --- 4. OOB check: if (idx < 0 || idx >= (ll)len) raise IndexError ---
    // V.3: built in IREP2, back-migrated for the legacy code_ifthenelset.
    const type2tc oob_llt = migrate_type(ll_type);
    expr2tc oob_idx, oob_len;
    migrate_expr(build_symbol(idx_sym), oob_idx);
    migrate_expr(build_typecast(build_symbol(len_sym), ll_type), oob_len);
    expr2tc still_neg = lessthan2tc(oob_idx, gen_zero(oob_llt));
    expr2tc idx_ge_len = greaterthanequal2tc(oob_idx, oob_len);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset oob_guard;
    oob_guard.cond() = migrate_expr_back(or2tc(still_neg, idx_ge_len));
    oob_guard.then_case() = throw_code;
    oob_guard.location() = loc;
    converter_.add_instruction(oob_guard);

    // --- 5. __python_str_slice(array, idx, idx+1, 1) ---
    // idx is now a normalized, in-bounds positive index; the slice helper
    // produces a fresh alloca'd single-char null-terminated string.
    exprt end_expr =
      build_add(build_symbol(idx_sym), from_integer(1, ll_type), ll_type);

    exprt slice_call = build_call_expr(
      get_str_slice_sym(),
      gen_pointer_type(char_type()),
      {array, build_symbol(idx_sym), end_expr, from_integer(1, ll_type)});
    slice_call.location() = loc;
    return slice_call;
  }

  // Handle static arrays
  return build_index(array, pos_expr, array.type().subtype());
}

exprt python_list::extract_pyobject_value(
  const exprt &pyobject_expr,
  const typet &elem_type,
  bool mixed_numeric,
  bool string_safe)
{
  // For float types, read __ESBMC_float_buf[item->float_idx].
  // This avoids the void*→integer truncation in --ir mode: float_idx is a size_t
  // (no sort mismatch in BV mode), and float_buf is a typed global double array
  // (real-sorted in --ir mode), so the array read gives the correct real value.
  if (elem_type.is_floatbv())
  {
    // Helper: build (*pyobject_expr).field for a given field/type.
    auto member_of = [&](const char *field, const typet &ftype) -> exprt {
      return build_deref_member(pyobject_expr, field, ftype);
    };

    // Look up __ESBMC_float_buf global (static in list.c, but still in symbol table)
    const symbolt *fbuf_sym =
      converter_.symbol_table().find_symbol("c:list.c@__ESBMC_float_buf");
    assert(fbuf_sym && "could not find __ESBMC_float_buf symbol");

    // Build __ESBMC_float_buf[item->float_idx]
    exprt float_idx_member = member_of("float_idx", size_type());
    exprt float_val =
      build_index(build_symbol(*fbuf_sym), float_idx_member, elem_type);

    if (!mixed_numeric)
      return float_val;

    // Dynamic index into a heterogeneous int/float list: the element may be an
    // int whose value lives behind item->value (not in float_buf), so reading
    // float_buf unconditionally would misread it (#5160). Dispatch on the
    // stored type_id: float elements come from float_buf, int elements are
    // promoted to double from their 8-byte payload. The float type_id is the
    // same hash the push path stamps onto float elements (float_type_id).
    const size_t float_type_id = std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(elem_type));

    // (double)*(long long *)item->value
    exprt value_member = member_of("value", pointer_typet(empty_typet()));
    exprt as_int_ptr =
      build_typecast(value_member, pointer_typet(long_long_int_type()));
    exprt int_val = build_dereference(as_int_ptr, long_long_int_type());
    exprt int_as_float = build_typecast(int_val, elem_type);

    // item->type_id == float_type_id ? float_buf[float_idx] : (double)int
    // V.3: built in IREP2 (both branches are elem_type, so the if2t types
    // agree), back-migrated at the return.
    const type2tc et2 = migrate_type(elem_type);
    expr2tc tid2, fv2, iaf2;
    migrate_expr(member_of("type_id", size_type()), tid2);
    migrate_expr(float_val, fv2);
    migrate_expr(int_as_float, iaf2);
    const expr2tc is_float =
      equality2tc(tid2, from_integer(float_type_id, migrate_type(size_type())));
    return migrate_expr_back(if2tc(et2, is_float, fv2, iaf2));
  }

  // Extract value from PyObject: (*pyobject_expr).value
  exprt obj_value =
    build_deref_member(pyobject_expr, "value", pointer_typet(empty_typet()));

  // Element whose static type was erased to any_type() (void*): this happens
  // when a dict/list crosses into an *unannotated* parameter, where keys()/
  // values() reads have no compile-time element type (#5444). The generic
  // "cast void* to T* and dereference" path below overruns a string element: a
  // str is stored as a short char[] at item->value (e.g. a 2-byte key), so an
  // 8-byte void* dereference reads past the array (array-bounds violation). The
  // element's *runtime* type is recorded in item->type_id (stamped at push
  // time), so when it marks a string, keep the stored pointer (== char*, the
  // same value the annotated str path returns) with no dereference. Every other
  // type keeps the proven dereference read, so numeric elements are unaffected.
  // The dispatch key is the caller-stamped type, not a usage heuristic, so this
  // stays sound. Gated on string_safe because the result is an if-expression
  // (rvalue): callers that take its address (membership/find on heaps, queues,
  // sets) need the plain dereference lvalue, so only value-context reads
  // (subscript) opt in.
  if (
    string_safe && elem_type.is_pointer() &&
    elem_type.subtype().id() == "empty")
  {
    const size_t str_type_id = std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(pointer_typet(char_type())));
    exprt type_id_member =
      build_deref_member(pyobject_expr, "type_id", size_type());
    // Default: cast void* to void** and dereference (the read that already
    // works for numeric / pointer-by-value elements).
    exprt as_default = build_dereference(
      build_typecast(obj_value, pointer_typet(elem_type)), elem_type);
    // item->type_id == str_type_id ? obj_value : *(T*)obj_value
    // V.3: built in IREP2 (both branches are elem_type/void*, so the if2t
    // types agree), back-migrated at the return. Mirrors the float-dispatch
    // if2t above.
    const type2tc et2 = migrate_type(elem_type);
    expr2tc tid2, ov2, def2;
    migrate_expr(type_id_member, tid2);
    migrate_expr(obj_value, ov2);
    migrate_expr(as_default, def2);
    const expr2tc is_str =
      equality2tc(tid2, from_integer(str_type_id, migrate_type(size_type())));
    return migrate_expr_back(if2tc(et2, is_str, ov2, def2));
  }

  // For array types, return pointer to element type instead of pointer to array
  // The dereference system doesn't support array types as target types
  if (elem_type.is_array())
  {
    const array_typet &arr_type = to_array_type(elem_type);
    // Cast to pointer to element type (e.g., char* instead of char[2]*)
    return build_typecast(obj_value, pointer_typet(arr_type.subtype()));
  }

  // For char* strings and None (_Bool*), the void* already contains the pointer value
  // For all other types, the void* contains a pointer to the value
  if (
    elem_type.is_pointer() &&
    (elem_type.subtype() == char_type() || elem_type.subtype() == bool_type()))
  {
    // String and None case: cast void* directly to the pointer type (no dereference needed)
    return build_typecast(obj_value, elem_type);
  }
  else
  {
    // All other types: cast void* to pointer-to-type, then dereference
    exprt tc = build_typecast(obj_value, pointer_typet(elem_type));
    return build_dereference(tc, elem_type);
  }
}
