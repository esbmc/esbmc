#include "python_list_internal.h"

using namespace python_expr;
using namespace python_list_detail;

exprt python_list::compare(
  const exprt &l1,
  const exprt &l2,
  const std::string &op)
{
  const symbolt *list_eq_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_eq");
  assert(list_eq_func_sym);

  // Convert member expressions into temporary symbols
  auto materialize_if_needed = [&](const exprt &e) -> exprt {
    if (e.id() == "member")
    {
      // Extract member expression to a temporary variable
      const member_exprt &member = to_member_expr(e);

      symbolt &temp_sym = converter_.create_tmp_symbol(
        list_value_, "$list_temp$", e.type(), exprt());

      code_declt temp_decl(build_symbol(temp_sym));
      temp_decl.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_decl);

      code_assignt temp_assign(build_symbol(temp_sym), member);
      temp_assign.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_assign);

      return build_symbol(temp_sym);
    }
    return e;
  };

  const exprt converted_l1 = materialize_if_needed(l1);
  const exprt converted_l2 = materialize_if_needed(l2);

  const symbolt *lhs_symbol =
    converter_.find_symbol(converted_l1.identifier().as_string());
  const symbolt *rhs_symbol =
    converter_.find_symbol(converted_l2.identifier().as_string());
  assert(lhs_symbol);
  assert(rhs_symbol);

  const bool lhs_is_set = lhs_symbol->is_set;
  const bool rhs_is_set = rhs_symbol->is_set;
  if (lhs_is_set || rhs_is_set)
  {
    if (!(lhs_is_set && rhs_is_set))
      return gen_boolean(op == "NotEq");

    // Python set ordering (< strict subset, <= subset-or-equal, and the > / >=
    // supersets) requires proper subset checking, which is not yet modelled.
    // Reject it explicitly rather than silently returning the Eq/NotEq result,
    // which would be wrong for these operators.
    if (op == "Lt" || op == "LtE" || op == "Gt" || op == "GtE")
      throw std::runtime_error(
        "set ordering (<, <=, >, >=) is not yet supported");

    symbolt *set_eq_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_set_eq");
    if (!set_eq_func)
    {
      symbolt new_symbol;
      new_symbol.name = "__ESBMC_list_set_eq";
      new_symbol.id = "c:@F@__ESBMC_list_set_eq";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet func_type;
      func_type.return_type() = bool_type();
      typet list_ptr = converter_.get_type_handler().get_list_type();
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      new_symbol.set_type(func_type);

      converter_.symbol_table().add(new_symbol);
      set_eq_func =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_set_eq");
    }

    locationt loc = converter_.get_location_from_decl(list_value_);
    symbolt &eq_ret = converter_.create_tmp_symbol(
      list_value_,
      "set_eq_tmp",
      bool_type(),
      migrate_expr_back(gen_false_expr())); // V.3
    code_declt eq_ret_decl(build_symbol(eq_ret));
    converter_.add_instruction(eq_ret_decl);

    code_function_callt set_eq_call;
    set_eq_call.function() = build_symbol(*set_eq_func);
    set_eq_call.lhs() = build_symbol(eq_ret);
    set_eq_call.arguments().push_back(
      lhs_symbol->get_type().is_pointer()
        ? build_symbol(*lhs_symbol)
        : build_address_of(build_symbol(*lhs_symbol)));
    set_eq_call.arguments().push_back(
      rhs_symbol->get_type().is_pointer()
        ? build_symbol(*rhs_symbol)
        : build_address_of(build_symbol(*rhs_symbol)));
    set_eq_call.type() = bool_type();
    set_eq_call.location() = loc;
    converter_.add_instruction(set_eq_call);

    // V.3: build `eq_ret == (op == "Eq")` in IREP2.
    expr2tc er2;
    migrate_expr(build_symbol(eq_ret), er2);
    return migrate_expr_back(
      equality2tc(er2, op == "Eq" ? gen_true_expr() : gen_false_expr()));
  }

  // Fast path for list equality/inequality when we have concrete type-map
  // entries for both operands. This avoids __ESBMC_list_eq loops.
  if (op == "Eq" || op == "NotEq")
  {
    auto resolve_map_id = [&](const symbolt *sym) -> std::string {
      const std::string direct_id = sym->id.as_string();
      auto has_map = [&](const std::string &id) {
        auto it = list_type_map.find(id);
        return it != list_type_map.end() && !it->second.empty();
      };

      if (has_map(direct_id))
        return direct_id;

      const exprt &v = sym->get_value();
      if (v.is_symbol())
      {
        const std::string alias_id = v.identifier().as_string();
        if (has_map(alias_id))
          return alias_id;
      }
      else if (
        v.id() == "typecast" && !v.operands().empty() && v.op0().is_symbol())
      {
        const std::string alias_id = v.op0().identifier().as_string();
        if (has_map(alias_id))
          return alias_id;
      }

      return direct_id;
    };

    const std::string lhs_id = resolve_map_id(lhs_symbol);
    const std::string rhs_id = resolve_map_id(rhs_symbol);
    auto is_numeric = [](const typet &t) {
      return t.is_signedbv() || t.is_unsignedbv() || t.is_floatbv();
    };
    auto is_bool = [](const typet &t) { return t == bool_type(); };

    auto is_concrete_map = [&](const std::string &list_id) -> bool {
      auto it = list_type_map.find(list_id);
      if (it == list_type_map.end() || it->second.empty())
        return false;

      for (const auto &entry : it->second)
      {
        if (entry.first.empty())
          return false;

        const symbolt *elem_sym = converter_.find_symbol(entry.first);
        if (!elem_sym)
          return false;
        if (!(is_numeric(elem_sym->get_type()) ||
              is_bool(elem_sym->get_type())))
          return false;
      }
      return true;
    };

    auto has_mixed_int_float = [&](const std::string &list_id) -> bool {
      auto it = list_type_map.find(list_id);
      if (it == list_type_map.end() || it->second.empty())
        return false;

      bool has_int = false;
      bool has_float = false;
      for (const auto &entry : it->second)
      {
        const symbolt *elem_sym = converter_.find_symbol(entry.first);
        const typet t = elem_sym ? elem_sym->get_type() : entry.second;
        if (t.is_floatbv())
          has_float = true;
        else if (t.is_signedbv() || t.is_unsignedbv())
          has_int = true;
      }
      return has_int && has_float;
    };

    // Concrete nested lists can be compared directly without forcing the
    // recursive list operational model. This keeps matrix-style comparisons
    // from blowing up into large __ESBMC_list_eq trees.
    const typet list_model_type = converter_.get_type_handler().get_list_type();
    std::function<bool(
      const symbolt *, const symbolt *, std::size_t, expr2tc &)>
      build_nested_equality;
    build_nested_equality = [&](
                              const symbolt *lhs_list,
                              const symbolt *rhs_list,
                              std::size_t depth,
                              expr2tc &result) -> bool {
      if (
        !lhs_list || !rhs_list ||
        depth >= static_cast<std::size_t>(get_list_compare_depth()))
        return false;

      const std::string lhs_list_id = resolve_map_id(lhs_list);
      const std::string rhs_list_id = resolve_map_id(rhs_list);
      auto lhs_it = list_type_map.find(lhs_list_id);
      auto rhs_it = list_type_map.find(rhs_list_id);
      if (lhs_it == list_type_map.end() || rhs_it == list_type_map.end())
        return false;

      const std::size_t lhs_size = lhs_it->second.size();
      const std::size_t rhs_size = rhs_it->second.size();
      if (lhs_size != rhs_size)
      {
        result = gen_false_expr();
        return true;
      }
      if (lhs_size > 64)
        return false;

      result = gen_true_expr();
      for (std::size_t i = 0; i < lhs_size; ++i)
      {
        const std::string lhs_elem_id = get_list_element_id(lhs_list_id, i);
        const std::string rhs_elem_id = get_list_element_id(rhs_list_id, i);
        const symbolt *lhs_elem_sym = converter_.find_symbol(lhs_elem_id);
        const symbolt *rhs_elem_sym = converter_.find_symbol(rhs_elem_id);
        if (!lhs_elem_sym || !rhs_elem_sym)
          return false;

        const typet &lhs_elem_type = lhs_elem_sym->get_type();
        const typet &rhs_elem_type = rhs_elem_sym->get_type();
        const bool lhs_is_list = lhs_elem_type == list_model_type;
        const bool rhs_is_list = rhs_elem_type == list_model_type;
        expr2tc elem_equal;

        if (lhs_is_list || rhs_is_list)
        {
          if (!(lhs_is_list && rhs_is_list))
            elem_equal = gen_false_expr();
          else
          {
            const std::string lhs_nested_id = resolve_map_id(lhs_elem_sym);
            const std::string rhs_nested_id = resolve_map_id(rhs_elem_sym);
            const symbolt *lhs_nested = converter_.find_symbol(lhs_nested_id);
            const symbolt *rhs_nested = converter_.find_symbol(rhs_nested_id);
            if (!build_nested_equality(
                  lhs_nested, rhs_nested, depth + 1, elem_equal))
              return false;
          }
        }
        else
        {
          if (
            !(is_numeric(lhs_elem_type) || is_bool(lhs_elem_type)) ||
            !(is_numeric(rhs_elem_type) || is_bool(rhs_elem_type)))
            return false;

          const exprt index = from_integer(BigInt(i), size_type());
          exprt lhs_at =
            build_list_at_call(build_symbol(*lhs_list), index, list_value_);
          exprt rhs_at =
            build_list_at_call(build_symbol(*rhs_list), index, list_value_);
          exprt lhs_value = extract_pyobject_value(lhs_at, lhs_elem_type);
          exprt rhs_value = extract_pyobject_value(rhs_at, rhs_elem_type);
          if (lhs_elem_type != rhs_elem_type)
          {
            if (!(is_numeric(lhs_elem_type) && is_numeric(rhs_elem_type)))
              return false;
            lhs_value = build_typecast(lhs_value, double_type());
            rhs_value = build_typecast(rhs_value, double_type());
          }

          expr2tc lhs_value2, rhs_value2;
          migrate_expr(lhs_value, lhs_value2);
          migrate_expr(rhs_value, rhs_value2);
          elem_equal = equality2tc(lhs_value2, rhs_value2);
        }

        result = and2tc(result, elem_equal);
      }
      return true;
    };

    const typet lhs_first_type = get_list_element_type(lhs_id, 0);
    const typet rhs_first_type = get_list_element_type(rhs_id, 0);
    if (lhs_first_type == list_model_type && rhs_first_type == list_model_type)
    {
      expr2tc nested_equal;
      if (build_nested_equality(lhs_symbol, rhs_symbol, 0, nested_equal))
      {
        if (op == "NotEq")
          nested_equal = not2tc(nested_equal);
        return migrate_expr_back(nested_equal);
      }
    }

    if (is_concrete_map(lhs_id) && is_concrete_map(rhs_id))
    {
      if (has_mixed_int_float(lhs_id) || has_mixed_int_float(rhs_id))
      {
        // For mixed int/float lists, runtime operations (e.g., sort) can
        // reorder heterogeneous elements and invalidate index->type mapping.
        // Keep the structural runtime comparator for soundness.
      }
      else
      {
        const size_t lhs_n = get_list_type_map_size(lhs_id);
        const size_t rhs_n = get_list_type_map_size(rhs_id);

        if (lhs_n == rhs_n && lhs_n <= 64)
        {
          expr2tc all_equal = gen_true_expr(); // V.3: accumulate in IREP2
          bool comparable = true;

          for (size_t i = 0; i < lhs_n; ++i)
          {
            const std::string lhs_elem_id = get_list_element_id(lhs_id, i);
            const std::string rhs_elem_id = get_list_element_id(rhs_id, i);
            const symbolt *lhs_elem_sym = converter_.find_symbol(lhs_elem_id);
            const symbolt *rhs_elem_sym = converter_.find_symbol(rhs_elem_id);
            const typet lhs_elem_type = lhs_elem_sym
                                          ? lhs_elem_sym->get_type()
                                          : get_list_element_type(lhs_id, i);
            const typet rhs_elem_type = rhs_elem_sym
                                          ? rhs_elem_sym->get_type()
                                          : get_list_element_type(rhs_id, i);
            if (lhs_elem_type.is_nil() || rhs_elem_type.is_nil())
            {
              comparable = false;
              break;
            }

            const exprt idx = from_integer(BigInt(i), size_type());
            exprt lhs_at =
              build_list_at_call(build_symbol(*lhs_symbol), idx, list_value_);
            exprt rhs_at =
              build_list_at_call(build_symbol(*rhs_symbol), idx, list_value_);
            exprt lhs_val = extract_pyobject_value(lhs_at, lhs_elem_type);
            exprt rhs_val = extract_pyobject_value(rhs_at, rhs_elem_type);

            // Same-type numeric/bool compares directly; mixed numeric promotes
            // both sides to double first. (V.3: built in IREP2.)
            if (
              lhs_elem_type == rhs_elem_type &&
              (is_numeric(lhs_elem_type) || is_bool(lhs_elem_type)))
            {
              // direct compare
            }
            else if (is_numeric(lhs_elem_type) && is_numeric(rhs_elem_type))
            {
              lhs_val = build_typecast(lhs_val, double_type());
              rhs_val = build_typecast(rhs_val, double_type());
            }
            else
            {
              comparable = false;
              break;
            }

            expr2tc lhs_val2, rhs_val2;
            migrate_expr(lhs_val, lhs_val2);
            migrate_expr(rhs_val, rhs_val2);
            all_equal = and2tc(all_equal, equality2tc(lhs_val2, rhs_val2));
          }

          if (comparable)
          {
            if (op == "NotEq")
              return migrate_expr_back(not2tc(all_equal));
            return migrate_expr_back(all_equal);
          }
        }
      }
    }
  }

  // ── Ordering operators: Lt, LtE, Gt, GtE ──────────────────────────────────
  // Implemented via __ESBMC_list_lt (lexicographic less-than):
  //   Lt  : list_lt(l1, l2)
  //   LtE : !list_lt(l2, l1)    (i.e. !(l1 > l2))
  //   Gt  : list_lt(l2, l1)
  //   GtE : !list_lt(l1, l2)    (i.e. !(l1 < l2))
  if (op == "Lt" || op == "LtE" || op == "Gt" || op == "GtE")
  {
    // Look up or register the __ESBMC_list_lt symbol.
    const symbolt *list_lt_func_sym =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_lt");
    if (!list_lt_func_sym)
    {
      symbolt new_symbol;
      new_symbol.name = "__ESBMC_list_lt";
      new_symbol.id = "c:@F@__ESBMC_list_lt";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet func_type;
      func_type.return_type() = bool_type();
      typet list_ptr = converter_.get_type_handler().get_list_type();
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(int_type()));
      func_type.arguments().push_back(code_typet::argumentt(size_type()));
      new_symbol.set_type(func_type);

      converter_.symbol_table().add(new_symbol);
      list_lt_func_sym =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_lt");
    }
    assert(list_lt_func_sym);

    // Determine element type flags from both lists and merge them so that
    // cross-type comparisons like [1,2] < [1.0,2.0] are handled correctly.
    int type_flag_lhs = 0, type_flag_rhs = 0;
    size_t float_type_id_lhs = 0, float_type_id_rhs = 0;
    get_list_type_flags(
      lhs_symbol->id.as_string(),
      converter_.get_type_handler(),
      type_flag_lhs,
      float_type_id_lhs);
    get_list_type_flags(
      rhs_symbol->id.as_string(),
      converter_.get_type_handler(),
      type_flag_rhs,
      float_type_id_rhs);

    const bool lhs_has_float = (type_flag_lhs == 1 || type_flag_lhs == 3);
    const bool lhs_has_int = (type_flag_lhs == 0 || type_flag_lhs == 3);
    const bool rhs_has_float = (type_flag_rhs == 1 || type_flag_rhs == 3);
    const bool rhs_has_int = (type_flag_rhs == 0 || type_flag_rhs == 3);
    const bool is_string = (type_flag_lhs == 2 || type_flag_rhs == 2);
    const bool has_float = lhs_has_float || rhs_has_float;
    const bool has_int = lhs_has_int || rhs_has_int;
    const size_t float_type_id =
      float_type_id_lhs ? float_type_id_lhs : float_type_id_rhs;

    int type_flag;
    if (is_string)
      type_flag = 2;
    else if (has_float && has_int)
      type_flag = 3;
    else if (has_float)
      type_flag = 1;
    else
      type_flag = 0;

    // Emit: lt_ret = __ESBMC_list_lt(a, b, type_flag, float_type_id)
    // Derivations (total order):
    //   Lt  : list_lt(l1, l2) == true   → no swap, check true
    //   LtE : !list_lt(l2, l1) == true  → swap,    check false
    //   Gt  : list_lt(l2, l1) == true   → swap,    check true
    //   GtE : !list_lt(l1, l2) == true  → no swap, check false
    const bool swap = (op == "LtE" || op == "Gt");
    const symbolt *a_sym = swap ? rhs_symbol : lhs_symbol;
    const symbolt *b_sym = swap ? lhs_symbol : rhs_symbol;

    symbolt &lt_ret = converter_.create_tmp_symbol(
      list_value_, "lt_tmp", bool_type(), migrate_expr_back(gen_false_expr()));
    code_declt lt_ret_decl(build_symbol(lt_ret));
    converter_.add_instruction(lt_ret_decl);

    code_function_callt lt_call;
    lt_call.function() = build_symbol(*list_lt_func_sym);
    lt_call.lhs() = build_symbol(lt_ret);
    lt_call.arguments().push_back(build_symbol(*a_sym));
    lt_call.arguments().push_back(build_symbol(*b_sym));
    lt_call.arguments().push_back(from_integer(type_flag, int_type()));
    lt_call.arguments().push_back(from_integer(float_type_id, size_type()));
    lt_call.type() = bool_type();
    lt_call.location() = converter_.get_location_from_decl(list_value_);
    converter_.add_instruction(lt_call);

    // Lt / Gt → lt_ret must be true; LtE / GtE → lt_ret must be false
    // V.3: build `lt_ret == (op is Lt/Gt)` in IREP2.
    expr2tc ltr2;
    migrate_expr(build_symbol(lt_ret), ltr2);
    const bool want_true = (op == "Lt" || op == "Gt");
    return migrate_expr_back(
      equality2tc(ltr2, want_true ? gen_true_expr() : gen_false_expr()));
  }

  // ── Equality operators: Eq, NotEq ─────────────────────────────────────────

  // Compute list type_id for nested list detection
  const typet &list_type = l1.type();
  const std::string list_type_name =
    converter_.get_type_handler().type_to_string(list_type);
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(list_type_name), config.ansi_c.address_width));

  symbolt &eq_ret = converter_.create_tmp_symbol(
    list_value_, "eq_tmp", bool_type(), migrate_expr_back(gen_false_expr()));
  code_declt eq_ret_decl(build_symbol(eq_ret));
  converter_.add_instruction(eq_ret_decl);

  // Get max depth from configuration option
  int max_depth = get_list_compare_depth();
  constant_exprt max_depth_expr(size_type());
  max_depth_expr.set_value(
    integer2binary(max_depth, config.ansi_c.address_width));

  // Merge the float element type_id from both operands so that mixed int/float
  // elements compare numerically (Python's 1 == 1.0), as list_lt already does.
  int type_flag_lhs = 0, type_flag_rhs = 0;
  size_t float_type_id_lhs = 0, float_type_id_rhs = 0;
  get_list_type_flags(
    lhs_symbol->id.as_string(),
    converter_.get_type_handler(),
    type_flag_lhs,
    float_type_id_lhs);
  get_list_type_flags(
    rhs_symbol->id.as_string(),
    converter_.get_type_handler(),
    type_flag_rhs,
    float_type_id_rhs);
  const size_t float_type_id =
    float_type_id_lhs ? float_type_id_lhs : float_type_id_rhs;

  // Statically-known element byte size for the primitive comparison, so the
  // model's __ESBMC_values_equal takes its branch-free fast path instead of
  // memcmp's symbolic-size byte loop (the dominant cost when comparing large
  // lists, e.g. `assert l == [...]`). Emitted only when both operands' first
  // element is the same fixed-width scalar; 0 otherwise, which makes the model
  // fall back to the per-element a->size read (exact prior behaviour).
  size_t eq_elem_size_bytes = 0;
  {
    auto scalar_width = [](const typet &t) -> size_t {
      if (
        (t.id() == "signedbv" || t.id() == "unsignedbv" ||
         t.id() == "floatbv" || t.id() == "fixedbv") &&
        !t.width().empty())
        return std::stoull(t.width().as_string(), nullptr, 10) / 8;
      return 0;
    };
    const typet lt =
      get_list_element_type(converted_l1.identifier().as_string(), 0);
    const typet rt =
      get_list_element_type(converted_l2.identifier().as_string(), 0);
    size_t lw = lt.is_nil() ? 0 : scalar_width(lt);
    size_t rw = rt.is_nil() ? 0 : scalar_width(rt);
    if (lw != 0 && lw == rw)
      eq_elem_size_bytes = lw;
  }

  code_function_callt list_eq_func_call;
  list_eq_func_call.function() = build_symbol(*list_eq_func_sym);
  list_eq_func_call.lhs() = build_symbol(eq_ret);
  // passing arguments
  list_eq_func_call.arguments().push_back(build_symbol(*lhs_symbol)); // l1
  list_eq_func_call.arguments().push_back(build_symbol(*rhs_symbol)); // l2
  list_eq_func_call.arguments().push_back(list_type_id);   // list_type_id
  list_eq_func_call.arguments().push_back(max_depth_expr); // max_depth
  list_eq_func_call.arguments().push_back(
    from_integer(float_type_id, size_type())); // float_type_id
  list_eq_func_call.arguments().push_back(
    from_integer(eq_elem_size_bytes, size_type())); // elem_size
  list_eq_func_call.type() = bool_type();
  list_eq_func_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(list_eq_func_call);

  // V.3: build `eq_ret == (op == "Eq")` in IREP2.
  expr2tc leqr2;
  migrate_expr(build_symbol(eq_ret), leqr2);
  return migrate_expr_back(
    equality2tc(leqr2, op == "Eq" ? gen_true_expr() : gen_false_expr()));
}

exprt python_list::contains(const exprt &item, const exprt &list)
{
  // Get type and size information for the item
  list_elem_info item_info = get_list_element_info(list_value_, item);

  // Find the list_contains function
  const symbolt *list_contains_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_contains");
  assert(list_contains_func);

  // Create a temporary variable to store the result
  symbolt &contains_ret = converter_.create_tmp_symbol(
    list_value_,
    "contains_tmp",
    bool_type(),
    migrate_expr_back(gen_false_expr())); // V.3
  code_declt contains_ret_decl(build_symbol(contains_ret));
  converter_.add_instruction(contains_ret_decl);

  // Build the function call as a statement
  code_function_callt contains_call;
  contains_call.function() = build_symbol(*list_contains_func);
  contains_call.lhs() = build_symbol(contains_ret);

  // Pass the list directly
  contains_call.arguments().push_back(list);

  // The model (__ESBMC_list_contains) compares *item against *elem->value, i.e.
  // it expects `item` to *point to* the search value. A string/None element is
  // stored as the pointer value itself in elem->value, so a char*/_Bool* item
  // is passed by value to match. But a user-class instance is stored as a
  // `Class*` slot (8 bytes), so the search value IS that pointer: passing the
  // `Class*` by value would make the model dereference it and compare the
  // pointee struct's first word instead, so `obj in [obj]` wrongly returns
  // False (#4805). Pass its address, like the value-type path, so *item
  // recovers the stored `Class*`.
  const typet &item_type = item_info.elem_symbol->get_type();
  exprt item_arg;
  if (item_type.is_pointer() && !converter_.is_user_class_pointer(item_type))
  {
    // String/None and other non-class pointers: pass the pointer value direct.
    item_arg = build_symbol(*item_info.elem_symbol);
  }
  else
  {
    // Value types and user-class references: take the address.
    item_arg = build_address_of(build_symbol(*item_info.elem_symbol));
  }

  contains_call.arguments().push_back(item_arg);

  // For void/char pointers from iteration, use stored type info from list
  exprt type_hash = build_symbol(*item_info.elem_type_sym);
  exprt elem_size = item_info.elem_size;

  // void* items (e.g. a loop variable over a string list) need the stored
  // char-array type_id and runtime length recovered from list_type_map.
  // The lookup is keyed by symbol name, so non-symbol receivers cannot carry
  // void* elements; skipping them is sound.
  if (
    item_info.elem_symbol->get_type() == pointer_typet(empty_typet()) &&
    list.is_symbol())
  {
    const std::string &list_name = list.identifier().as_string();
    auto type_map_it = list_type_map.find(list_name);

    if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
    {
      // Look for a string array type (char array) in the list
      for (const auto &stored_entry : type_map_it->second)
      {
        const typet &stored_type = stored_entry.second;

        // Check if stored type is a char array (string)
        if (stored_type.is_array() && stored_type.subtype() == char_type())
        {
          // Use the stored string array type instead of void pointer type
          const type_handler type_handler_ = converter_.get_type_handler();
          const std::string stored_type_name =
            type_handler_.type_to_string(stored_type);

          constant_exprt stored_hash(size_type());
          stored_hash.set_value(integer2binary(
            std::hash<std::string>{}(stored_type_name),
            config.ansi_c.address_width));
          type_hash = stored_hash;

          // Use strlen for void* strings from iteration
          const symbolt *strlen_symbol =
            converter_.symbol_table().find_symbol("c:@F@strlen");
          if (strlen_symbol)
          {
            // Call strlen to get actual string length
            symbolt &strlen_result = converter_.create_tmp_symbol(
              list_value_,
              "$strlen_result$",
              size_type(),
              gen_zero(size_type()));
            code_declt strlen_decl(build_symbol(strlen_result));
            strlen_decl.location() = item_info.location;
            converter_.add_instruction(strlen_decl);

            code_function_callt strlen_call;
            strlen_call.function() = build_symbol(*strlen_symbol);
            strlen_call.lhs() = build_symbol(strlen_result);
            strlen_call.arguments().push_back(
              build_symbol(*item_info.elem_symbol));
            strlen_call.type() = size_type();
            strlen_call.location() = item_info.location;
            converter_.add_instruction(strlen_call);

            // Add 1 for null terminator: size = strlen(s) + 1. strlen_result
            // is a synthetic size_type symbol, so build it in IREP2 (V.3).
            elem_size = build_add(
              build_symbol(strlen_result),
              from_integer(1, strlen_result.get_type()),
              strlen_result.get_type());
          }

          break; // Found string array type, use it
        }
      }
    }
  }

  contains_call.arguments().push_back(type_hash);
  contains_call.arguments().push_back(elem_size);

  contains_call.type() = bool_type();
  contains_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(contains_call);

  // V.3: build `contains_ret == true` in IREP2.
  expr2tc cr2;
  migrate_expr(build_symbol(contains_ret), cr2);
  exprt result = migrate_expr_back(equality2tc(cr2, gen_true_expr()));

  return result;
}

exprt python_list::build_min_max_for_mixed_numeric(
  const exprt &list_arg,
  const std::string &list_id,
  const std::string &func_name,
  irep_idt comparison_op)
{
  const TypeInfo &type_info = list_type_map.at(list_id);
  size_t n = type_info.size();

  if (n == 0)
    throw std::runtime_error(func_name + "() arg is an empty sequence");

  pointer_typet obj_ptr_type(
    converter_.get_type_handler().get_list_element_type());
  const typet double_t = double_type();
  locationt loc = converter_.get_location_from_decl(list_value_);

  // Declare a temp symbol, emit its declaration, and return its symbol_expr.
  auto make_tmp = [&](const std::string &name, const typet &type) -> exprt {
    symbolt &sym =
      converter_.create_tmp_symbol(list_value_, name, type, exprt());
    code_declt decl(build_symbol(sym));
    decl.location() = loc;
    converter_.add_instruction(decl);
    return build_symbol(sym);
  };

  // Access element i from the list and return it promoted to double.
  auto get_elem_as_double = [&](size_t i) -> exprt {
    typet orig_type = type_info[i].second;
    exprt list_at = build_list_at_call(
      list_arg, from_integer(BigInt(i), size_type()), list_value_);
    exprt obj = make_tmp("$list_obj$", obj_ptr_type);
    code_assignt assign_obj(obj, list_at);
    assign_obj.location() = loc;
    converter_.add_instruction(assign_obj);
    exprt val = extract_pyobject_value(obj, orig_type);
    return orig_type.is_floatbv() ? val : build_typecast(val, double_t);
  };

  exprt result = make_tmp("$minmax_result$", double_t);
  code_assignt init(result, get_elem_as_double(0));
  init.location() = loc;
  converter_.add_instruction(init);

  for (size_t i = 1; i < n; i++)
  {
    exprt elem = make_tmp("$minmax_elem$", double_t);
    code_assignt assign_elem(elem, get_elem_as_double(i));
    assign_elem.location() = loc;
    converter_.add_instruction(assign_elem);

    exprt condition(comparison_op, bool_type());
    condition.copy_to_operands(elem, result);
    code_ifthenelset ite;
    ite.cond() = condition;
    ite.then_case() = code_assignt(result, elem);
    ite.location() = loc;
    converter_.add_instruction(ite);
  }

  return result;
}

// list.count(x) / list.index(x): both pass (list, &value, type_id, size) to a
// C model that walks the list comparing elements (same plumbing as remove), and
// return a size_t value. `func_id` selects the model; the only difference is
// index() asserts the element is present (handled inside the model).
exprt python_list::build_count_index_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem,
  const std::string &func_id)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *func = converter_.symbol_table().find_symbol(func_id);
  if (!func)
    throw std::runtime_error(func_id + " function not found in symbol table");

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  // Args: (list, &value-or-ptr, type_id, size) — same shape as the remove call.
  exprt call = build_call_expr(
    *func,
    size_type(),
    {build_symbol(list),
     element_arg,
     build_symbol(*elem_info.elem_type_sym),
     elem_info.elem_size});
  call.location() = elem_info.location;

  return call;
}

exprt python_list::build_count_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  return build_count_index_list_call(list, op, elem, "c:@F@__ESBMC_list_count");
}

exprt python_list::build_index_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  // list.index(x): locate x with the non-asserting try_find_index (returns
  // SIZE_MAX when absent) and raise a ValueError from the frontend on the
  // not-found path, so `try/except ValueError` can catch it — instead of the
  // asserting __ESBMC_list_index model, which fires an uncatchable property
  // violation. Mirrors the dict KeyError and list.remove ValueError lowerings.
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *find_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_try_find_index");
  if (!find_func)
    throw std::runtime_error(
      "__ESBMC_list_try_find_index function not found in symbol table");

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  symbolt &idx = converter_.create_tmp_symbol(
    op, "index_ret", size_type(), gen_zero(size_type()));
  code_declt idx_decl(build_symbol(idx));
  idx_decl.location() = elem_info.location;
  converter_.add_instruction(idx_decl);

  code_function_callt find_call;
  find_call.function() = build_symbol(*find_func);
  find_call.lhs() = build_symbol(idx);
  find_call.arguments().push_back(build_symbol(list));
  find_call.arguments().push_back(element_arg);
  find_call.arguments().push_back(build_symbol(*elem_info.elem_type_sym));
  find_call.arguments().push_back(elem_info.elem_size);
  find_call.type() = size_type();
  find_call.location() = elem_info.location;
  converter_.add_instruction(find_call);

  // if (idx == SIZE_MAX) raise ValueError("list.index(x): x not in list")
  const BigInt size_max_val = power(2, bv_width(size_type())) - 1;
  const constant_exprt size_max(size_max_val, size_type());
  expr2tc idx2, max2;
  migrate_expr(build_symbol(idx), idx2);
  migrate_expr(size_max, max2);
  exprt not_found = migrate_expr_back(equality2tc(idx2, max2));

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ValueError", "list.index(x): x not in list");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);

  code_ifthenelset guard;
  guard.cond() = not_found;
  guard.then_case() = throw_code;
  guard.location() = elem_info.location;
  converter_.add_instruction(guard);

  return build_symbol(idx);
}

exprt python_list::build_index_range_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem,
  const exprt &start,
  const exprt &end)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_index_range");
  if (!func)
    throw std::runtime_error(
      "__ESBMC_list_index_range function not found in symbol table");

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  // Args: (list, &value-or-ptr, type_id, size, start, end). The start/end
  // Python ints are cast to the model's int64_t parameters by the call.
  exprt call = build_call_expr(
    *func,
    size_type(),
    {build_symbol(list),
     element_arg,
     build_symbol(*elem_info.elem_type_sym),
     elem_info.elem_size,
     start,
     end});
  call.location() = elem_info.location;

  return call;
}
