#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <util/std_code.h>
#include <util/symbolic_types.h>
#include <string>

std::unordered_map<std::string, std::vector<std::pair<std::string, typet>>>
  python_list::list_type_map{};

exprt python_list::build_push_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  const type_handler type_handler_ = converter_.get_type_handler();

  const std::string elem_type_name = type_handler_.type_to_string(elem.type());

  typet type_name_t =
    type_handler_.build_array(char_type(), elem_type_name.size() + 1);

  std::vector<unsigned char> type_name_str(
    elem_type_name.begin(), elem_type_name.end());
  type_name_str.push_back('\0');

  exprt type_name = converter_.make_char_array_expr(type_name_str, type_name_t);

  // Add a tmp variable to hold the type name as str
  symbolt &tmp_list_elem_type_symbol = converter_.create_tmp_symbol(
    op, "$list_elem_type$", size_type(), type_name);

  code_declt tmp_list_elem_type_decl(symbol_expr(tmp_list_elem_type_symbol));
  tmp_list_elem_type_decl.location() = converter_.get_location_from_decl(op);
  converter_.add_instruction(tmp_list_elem_type_decl);

  const symbolt *type_hash_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_hash_string");
  assert(type_hash_func_sym);

  // Initialize list_elem_type with return from hash function
  code_function_callt list_type_hash_func_call;
  list_type_hash_func_call.function() = symbol_expr(*type_hash_func_sym);
  list_type_hash_func_call.arguments().push_back(
    converter_.get_array_base_address(type_name)); // &type_name[0]
  list_type_hash_func_call.lhs() = symbol_expr(tmp_list_elem_type_symbol);
  list_type_hash_func_call.type() = size_type();
  list_type_hash_func_call.location() = converter_.get_location_from_decl(op);

  // Add list_hash_string call to the block
  converter_.add_instruction(list_type_hash_func_call);

  // 4.2.1 Build tmp variable for list element
  symbolt &tmp_list_elem_symbol =
    converter_.create_tmp_symbol(op, "$list_elem$", elem.type(), elem);

  // 4.2.2 Add tmp variable for list element in the block
  code_declt tmp_list_elem_decl(symbol_expr(tmp_list_elem_symbol));
  tmp_list_elem_decl.copy_to_operands(elem);
  tmp_list_elem_decl.location() = converter_.get_location_from_decl(op);
  //current_block->copy_to_operands(tmp_list_elem_decl);
  converter_.add_instruction(tmp_list_elem_decl);

  // Get push function symbol
  const symbolt *list_push_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_push");
  assert(list_push_func_sym);

  size_t list_elem_size = 0;
  try
  {
    if (tmp_list_elem_symbol.type.is_array())
    {
      size_t subtype_size =
        std::stoull(elem.type().subtype().width().as_string(), nullptr, 10);

      const array_typet &arr_type =
        static_cast<const array_typet &>(tmp_list_elem_symbol.type);

      list_elem_size =
        std::stoull(arr_type.size().value().as_string(), nullptr, 2) *
        subtype_size / 8;
    }
    else
    {
      list_elem_size =
        std::stoull(
          tmp_list_elem_symbol.type.width().as_string(), nullptr, 10) /
        8;
    }
  }
  catch (std::invalid_argument &)
  {
    list_elem_size = 1;
  }

  assert(list_elem_size);
  exprt e_size = from_integer(BigInt(list_elem_size), size_type());

  // 4.2.3 Add push function call
  code_function_callt list_push_func_call;
  list_push_func_call.function() = symbol_expr(*list_push_func_sym);
  // passing arguments
  list_push_func_call.arguments().push_back(symbol_expr(list)); // l
  list_push_func_call.arguments().push_back(
    address_of_exprt(symbol_expr(tmp_list_elem_symbol))); // &var
  list_push_func_call.arguments().push_back(
    symbol_expr(tmp_list_elem_type_symbol));         // type hash
  list_push_func_call.arguments().push_back(e_size); // sizeof(value_to_append)

  list_push_func_call.type() = bool_type();
  list_push_func_call.location() = converter_.get_location_from_decl(op);

  return list_push_func_call;
}

symbolt &python_list::create_list()
{
  array_typet inf_array_type(
    converter_.get_type_handler().get_list_element_type(),
    exprt("infinity", size_type()));

  // Build infinity array symbol
  exprt inf_array_value =
    gen_zero(get_complete_type(inf_array_type, converter_.ns), true);

  symbolt &inf_array_symbol = converter_.create_tmp_symbol(
    list_value_, "$storage$", inf_array_type, inf_array_value);

  inf_array_symbol.value.zero_initializer(true);
  inf_array_symbol.static_lifetime = true;

  // Add infinity array declaration to the block
  code_declt inf_array_decl(symbol_expr(inf_array_symbol));
  inf_array_decl.location() = converter_.get_location_from_decl(list_value_);
  //current_block->copy_to_operands(inf_array_decl);
  converter_.add_instruction(inf_array_decl);

  // Add List declaration

  // Build list symbol
  typet list_type = converter_.get_type_handler().get_list_type();

  symbolt &list_symbol =
    converter_.create_tmp_symbol(list_value_, "$list$", list_type, exprt());

  // Add list declaration to the block
  code_declt list_decl(symbol_expr(list_symbol));
  list_decl.location() = converter_.get_location_from_decl(list_value_);
  //current_block->copy_to_operands(list_decl);
  converter_.add_instruction(list_decl);

  // Build call to initialise the list with the infinity array
  const symbolt *list_create_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_create");
  assert(list_create_func_sym);

  code_function_callt list_create_func_call;
  list_create_func_call.function() = symbol_expr(*list_create_func_sym);
  list_create_func_call.lhs() = symbol_expr(list_symbol);
  list_create_func_call.arguments().push_back(
    converter_.get_array_base_address(symbol_expr(inf_array_symbol)));
  list_create_func_call.type() = list_type;
  list_create_func_call.location() =
    converter_.get_location_from_decl(list_value_);

  // Add list_create call to the block
  converter_.add_instruction(list_create_func_call);

  return list_symbol;
}

exprt python_list::get()
{
  symbolt &list_symbol = create_list();
  const std::string &list_id = list_symbol.id.as_string();

  for (auto &e : list_value_["elts"])
  {
    exprt elem = converter_.get_expr(e);
    exprt list_push_func_call =
      build_push_list_call(list_symbol, list_value_, elem);

    converter_.add_instruction(list_push_func_call);

    list_type_map[list_id].push_back(
      std::make_pair(elem.identifier().as_string(), elem.type()));
  }

  return symbol_expr(list_symbol);
}

exprt python_list::build_list_at_call(
  const exprt &list,
  const exprt &index,
  const nlohmann::json &element)
{
  pointer_typet obj_type(converter_.get_type_handler().get_list_element_type());

  const symbolt *list_at_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_at");
  assert(list_at_func_sym);

  side_effect_expr_function_callt list_at_call;
  list_at_call.function() = symbol_expr(*list_at_func_sym);
  if (list.type().is_pointer())
    list_at_call.arguments().push_back(list); // &l
  else
    list_at_call.arguments().push_back(address_of_exprt(list)); // &l

  list_at_call.arguments().push_back(index);
  list_at_call.type() = obj_type;
  list_at_call.location() = converter_.get_location_from_decl(element);

  return list_at_call;
}

exprt python_list::slice(const exprt &array, const nlohmann::json &slice_node)
{
  if (slice_node["_type"] == "Slice") // arr[lower:upper]
  {
    symbolt &sliced_list = create_list();

    exprt lower_expr = converter_.get_expr(slice_node["lower"]);
    exprt upper_expr = converter_.get_expr(slice_node["upper"]);

    // int counter = lower
    symbolt &counter = converter_.create_tmp_symbol(
      list_value_, "counter", size_type(), lower_expr);

    code_assignt counter_code(symbol_expr(counter), lower_expr);
    //current_block->copy_to_operands(counter_code);
    converter_.add_instruction(counter_code);

    // Build conditional for while loop (counter < upper)
    exprt cond("<", bool_type());
    cond.operands().push_back(symbol_expr(counter));
    cond.operands().push_back(upper_expr);

    // Build block with lish_push() calls and counter increment
    code_blockt then;

    // list_at call to get the element to insert
    exprt list_at_call =
      build_list_at_call(array, symbol_expr(counter), list_value_);

    const symbolt &list_at_ret = converter_.create_tmp_symbol(
      list_value_,
      "tmp_list_at",
      pointer_typet(converter_.get_type_handler().get_list_element_type()),
      exprt());

    code_declt tmp_list_at(symbol_expr(list_at_ret));
    tmp_list_at.copy_to_operands(list_at_call);
    then.copy_to_operands(tmp_list_at);

    // call list_push_object to insert the retrieved object
    const symbolt *list_push_object_func_sym =
      converter_.symbol_table().find_symbol("c:list.c@F@list_push_object");
    assert(list_push_object_func_sym);

    side_effect_expr_function_callt list_push_object_call;
    list_push_object_call.function() = symbol_expr(*list_push_object_func_sym);
    list_push_object_call.arguments().push_back(symbol_expr(sliced_list)); // &l
    list_push_object_call.arguments().push_back(symbol_expr(list_at_ret));
    list_push_object_call.type() = bool_type();
    list_push_object_call.location() =
      converter_.get_location_from_decl(list_value_);

    then.copy_to_operands(
      converter_.convert_expression_to_code(list_push_object_call));

    // increment counter
    exprt incr("+");
    incr.copy_to_operands(symbol_expr(counter));
    incr.copy_to_operands(gen_one(int_type()));
    code_assignt update(symbol_expr(counter), incr);
    then.copy_to_operands(update);

    // add while block for list_push() calls
    codet while_cod;
    while_cod.set_statement("while");
    while_cod.copy_to_operands(cond, then);
    //current_block->copy_to_operands(while_cod);
    converter_.add_instruction(while_cod);

    const auto &list_node = json_utils::get_var_value(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());

    for (size_t i = slice_node["lower"]["value"].get<size_t>();
         i < slice_node["upper"]["value"].get<size_t>();
         ++i)
    {
      exprt elt = converter_.get_expr(list_node["value"]["elts"][i]);

      list_type_map[sliced_list.id.as_string()].push_back(
        std::make_pair(elt.identifier().as_string(), elt.type()));
    }

    return symbol_expr(sliced_list);
  }
  else
  {
    nlohmann::json list_node;
    if (list_value_["value"].contains("id"))
    {
      list_node = json_utils::find_var_decl(
        list_value_["value"]["id"],
        converter_.current_function_name(),
        converter_.ast());
    }

    exprt pos = converter_.get_expr(slice_node);
    size_t index = 0;

    if (pos.type().is_array())
    {
      locationt l = converter_.get_location_from_decl(list_value_);
      throw std::runtime_error(
        "TypeError at " + l.get_file().as_string() + " " +
        l.get_line().as_string() +
        ": list indices must be integers or slices, not str");
    }

    // Adjust negative indexes
    if (slice_node.contains("op") && slice_node["op"]["_type"] == "USub")
    {
      if (list_node.is_null() || list_node["value"]["_type"] != "List")
      {
        BigInt v = binary2integer(pos.op0().value().c_str(), true);
        v *= -1;

        const array_typet &t = static_cast<const array_typet &>(array.type());
        BigInt s = binary2integer(t.size().value().c_str(), true);

        v += s;
        pos = from_integer(v, pos.type());
      }
      else
      {
        index = slice_node["operand"]["value"].get<size_t>();
        index = list_node["value"]["elts"].size() - index;
        pos = from_integer(index, size_type());
      }
    }
    else if (slice_node["_type"] == "Constant")
    {
      index = slice_node["value"].get<size_t>();
    }

    // lists are modelled as tag-struct __anon_typedef_List
    if (array.type().is_symbol() || array.type().subtype().is_symbol())
    {
      typet list_elem_type;

      if (
        array.type() == converter_.get_type_handler()
                          .get_list_type()) // Handle arrays of arrays
      {
        const auto &key = array.identifier().as_string();
        auto it = list_type_map.find(key);
        if (it != list_type_map.end())
        {
          if (index < it->second.size())
          {
            std::string &arr_elem_id = it->second.at(index).first;
            list_elem_type = it->second.at(index).second;

            if (list_elem_type == converter_.get_type_handler().get_list_type())
            {
              symbolt *l = converter_.find_symbol(arr_elem_id);
              assert(l);
              return symbol_expr(*l);
            }
          }
        }
      }

      if (list_elem_type == typet() && list_node.is_null())
      {
        // Handle case where list_node is not found - use default element type
        list_elem_type = converter_.get_type_handler().get_list_element_type();
      }
      else if (list_node["_type"] == "arg")
      {
        list_elem_type = converter_.get_type_handler().get_typet(
          list_node["annotation"]["slice"]["id"].get<std::string>());
      }
      else if (
        slice_node["_type"] == "Constant" || slice_node["_type"] == "BinOp" ||
        (slice_node["_type"] == "UnaryOp" &&
         slice_node["operand"]["_type"] == "Constant"))
      {
        const std::string &list_name = array.identifier().as_string();
        if (list_type_map[list_name].empty())
        {
          /* (Bruno): The referenced variable points to a list whose type map hasn’t been
	           * resolved yet (e.g., for function parameters). In this case, fall back
	           * to the node’s annotation. */
          const nlohmann::json list_value_node = json_utils::get_var_value(
            list_value_["value"]["id"],
            converter_.current_function_name(),
            converter_.ast());

          list_elem_type = converter_.get_type_handler().get_typet(
            list_value_node["annotation"]["slice"]["id"].get<std::string>());
        }
        else
        {
          size_t i = index;

          /* For list-multiplication initializations (e.g., [1] * f), we can
	           * simply use the type of the first element for now.*/
          if (!list_node.is_null() && list_node["value"]["_type"] == "BinOp")
            i = 0;

          try
          {
            list_elem_type = list_type_map[list_name].at(i).second;
          }
          catch (const std::out_of_range &)
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }
      else if (slice_node["_type"] == "Name") // Handling slicing with variables
      {
        if (!list_node.is_null() && list_node["_type"] == "arg")
        {
          list_elem_type = converter_.get_type_handler().get_typet(
            list_node["annotation"]["slice"]["id"].get<std::string>());
        }
        else
        {
          // Handle case where we need to find the variable declaration
          while (!list_node.is_null() &&
                 (!list_node.contains("value") ||
                  !list_node["value"].contains("elts") ||
                  !list_node["value"]["elts"].is_array()))
          {
            if (
              list_node.contains("value") && list_node["value"].contains("id"))
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
            list_elem_type = converter_.get_type_handler().get_typet(
              list_node["annotation"]["slice"]["id"].get<std::string>());
          }
          else if (list_elem_type == typet() && list_node.contains("value"))
          {
            list_elem_type =
              converter_
                .get_expr(json_utils::get_list_element(list_node["value"], 0))
                .type();
          }
        }
      }

      assert(pos != exprt());
      assert(list_elem_type != typet());

      // Build list_at() call
      exprt list_at_call = build_list_at_call(array, pos, list_value_);

      // Get obj->value and cast it to the correct type
      member_exprt obj_value(
        list_at_call, "value", pointer_typet(empty_typet()));

      {
        exprt &base = obj_value.struct_op();
        exprt deref("dereference");
        deref.type() = base.type().subtype();
        deref.move_to_operands(base);
        base.swap(deref);
      }

      // Direct typecast from obj->value (which is void*) to target type pointer
      typecast_exprt tc(obj_value, pointer_typet(list_elem_type));

      // Dereference to get the actual value
      dereference_exprt deref(list_elem_type);
      deref.op0() = tc;
      return deref;
    }

    // Handling static arrays

    typet t = array.type().subtype();
    return index_exprt(array, pos, t);
  }
  return exprt();
}

exprt python_list::compare(
  const exprt &l1,
  const exprt &l2,
  const std::string &op)
{
  const symbolt *list_eq_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_eq");
  assert(list_eq_func_sym);

  const symbolt *lhs_symbol =
    converter_.find_symbol(l1.identifier().as_string());
  const symbolt *rhs_symbol =
    converter_.find_symbol(l2.identifier().as_string());
  assert(lhs_symbol);
  assert(rhs_symbol);

  symbolt &eq_ret = converter_.create_tmp_symbol(
    list_value_, "eq_tmp", bool_type(), gen_boolean(false));
  code_declt eq_ret_decl(symbol_expr(eq_ret));
  converter_.add_instruction(eq_ret_decl);

  code_function_callt list_eq_func_call;
  list_eq_func_call.function() = symbol_expr(*list_eq_func_sym);
  list_eq_func_call.lhs() = symbol_expr(eq_ret);
  // passing arguments
  list_eq_func_call.arguments().push_back(symbol_expr(*lhs_symbol)); // l1
  list_eq_func_call.arguments().push_back(symbol_expr(*rhs_symbol)); // l2
  list_eq_func_call.type() = bool_type();
  list_eq_func_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(list_eq_func_call);

  //return list_eq_func_call;
  exprt cond("=", bool_type());
  cond.copy_to_operands(symbol_expr(eq_ret));
  if (op == "Eq")
    cond.copy_to_operands(gen_boolean(true));
  else
    cond.copy_to_operands(gen_boolean(false));

  return cond;
}

exprt python_list::create_vla(
  const nlohmann::json &element,
  const symbolt *list,
  symbolt *size_var,
  const exprt &list_elem)
{
  // Add counter for while loop
  symbolt &counter = converter_.create_tmp_symbol(
    element, "counter", int_type(), gen_zero(int_type()));

  code_assignt counter_code(symbol_expr(counter), gen_zero(int_type()));
  converter_.add_instruction(counter_code);

  // Build conditional for while loop (counter < len(arr))
  exprt cond("<", bool_type());
  cond.operands().push_back(symbol_expr(counter));
  cond.operands().push_back(symbol_expr(*size_var));

  // Build block with lish_push() calls and counter increment
  code_blockt then;
  exprt list_push_call = build_push_list_call(*list, element, list_elem);
  then.copy_to_operands(list_push_call);

  // increment counter
  exprt incr("+");
  incr.copy_to_operands(symbol_expr(counter));
  incr.copy_to_operands(gen_one(int_type()));
  code_assignt update(symbol_expr(counter), incr);
  then.copy_to_operands(update);

  // add while block for list_push() calls
  codet while_cod;
  while_cod.set_statement("while");
  while_cod.copy_to_operands(cond, then);
  converter_.add_instruction(while_cod);

  return symbol_expr(*list);
}

exprt python_list::list_repetition(
  const nlohmann::json &left_node,
  const nlohmann::json &right_node,
  const exprt &lhs,
  const exprt &rhs)
{
  typet list_type = converter_.get_type_handler().get_list_type();

  BigInt list_size;
  exprt list_elem;

  // Get list size from lhs (e.g.: 3 * [1])
  if (lhs.type() != list_type)
  {
    if (lhs.is_symbol())
    {
      symbolt *size_var = converter_.find_symbol(
        to_symbol_expr(lhs).get_identifier().as_string());
      assert(size_var);
      list_size = std::stoi(size_var->value.value().as_string(), nullptr, 2);
    }
    else if (lhs.is_constant())
      list_size = std::stoi(lhs.value().as_string(), nullptr, 2);

    // List element is the rhs
    list_elem = converter_.get_expr(right_node["elts"][0]);
  }

  // Get list size from rhs (e.g.: [1] * 3)
  if (rhs.type() != list_type)
  {
    // List element is the rhs
    list_elem = converter_.get_expr(left_node["elts"][0]);

    if (rhs.is_symbol()) // (e.g.: [1] * n)
    {
      symbolt *size_var = converter_.find_symbol(
        to_symbol_expr(rhs).get_identifier().as_string());

      assert(size_var);

      symbolt *list_symbol =
        converter_.find_symbol(lhs.identifier().as_string());
      assert(list_symbol);

      if (size_var->value.is_code())
      {
        return create_vla(list_value_, list_symbol, size_var, list_elem);
      }

      list_size = std::stoi(size_var->value.value().as_string(), nullptr, 2);
    }
    else if (rhs.is_constant())
      list_size = std::stoi(rhs.value().as_string(), nullptr, 2);
  }

  symbolt *list_symbol = converter_.find_symbol(lhs.identifier().as_string());
  assert(list_symbol);

  const std::string &list_id = converter_.current_lhs->identifier().as_string();

  for (int64_t i = 0; i < list_size.to_int64() - 1; ++i)
  {
    converter_.add_instruction(
      build_push_list_call(*list_symbol, list_value_, list_elem));

    list_type_map[list_id].push_back(
      std::make_pair(list_elem.identifier().as_string(), list_elem.type()));
  }

  return symbol_expr(*list_symbol);
}
