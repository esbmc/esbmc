#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_class_builder.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>

#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>

exprt python_converter::make_enum_member_struct_expr(
  const symbolt &int_sym,
  const std::string &class_name,
  const std::string &member_name)
{
  // Get the struct type for the enum class (e.g. tag-TrafficLight)
  const symbolt *type_sym = symbol_table_.find_symbol("tag-" + class_name);
  if (!type_sym || !type_sym->get_type().is_struct())
  {
    log_error("Enum class '{}' has no struct type in symbol table", class_name);
    abort();
  }

  const struct_typet &st = to_struct_type(type_sym->get_type());
  if (st.components().size() < 2)
  {
    log_error(
      "Enum class '{}' struct has fewer than 2 components (expected 'value' "
      "and 'name')",
      class_name);
    abort();
  }

  // Create (or reuse) a static char-array symbol for the member name string.
  const std::string str_id =
    "py:" + current_python_file + "@C@" + class_name + "@_name_" + member_name;
  if (!symbol_table_.find_symbol(str_id))
  {
    exprt str_val = string_builder_->build_string_literal(member_name);
    symbolt str_sym;
    str_sym.id = str_id;
    str_sym.name = "_name_" + member_name;
    str_sym.get_type() = str_val.type();
    str_sym.get_value() = str_val;
    str_sym.static_lifetime = true;
    str_sym.is_extern = false;
    str_sym.file_local = true;
    symbol_table_.add(str_sym);
  }
  const symbolt *str_sym = symbol_table_.find_symbol(str_id);
  assert(str_sym);

  // Build struct_exprt { value: int_sym, name: &str_sym[0] }
  struct_exprt struct_val(st);

  // value component: the integer value of the enum member
  struct_val.operands().push_back(symbol_expr(int_sym));

  // name component: char* pointer to the first element of the name string
  exprt str_expr = symbol_expr(*str_sym);
  exprt zero_idx = from_integer(0, index_type());
  exprt name_ptr = address_of_exprt(index_exprt(str_expr, zero_idx));
  name_ptr.type() = gen_pointer_type(char_type());
  struct_val.operands().push_back(name_ptr);

  return struct_val;
}

std::string
python_converter::extract_class_name_from_tag(const std::string &tag_name)
{
  if (tag_name.size() > 4 && tag_name.substr(0, 4) == "tag-")
    return tag_name.substr(4);
  return tag_name;
}

std::string
python_converter::create_normalized_self_key(const std::string &class_tag)
{
  std::string class_name = extract_class_name_from_tag(class_tag);
  return "self@" + class_name;
}

typet python_converter::clean_attribute_type(const typet &attr_type)
{
  typet clean_type = attr_type;
  clean_type.remove("#member_name");
  clean_type.remove("#location");
  clean_type.remove("#identifier");
  return clean_type;
}

exprt python_converter::create_member_expression(
  const symbolt &symbol,
  const std::string &attr_name,
  const typet &attr_type)
{
  typet clean_type = clean_attribute_type(attr_type);
  exprt source = symbol_exprt(symbol.id, symbol.get_type());
  if (source.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = source.type().subtype();
    deref.move_to_operands(source);
    source = std::move(deref);
  }
  typet source_type = source.type();
  if (source_type.id() == "symbol")
    source_type = ns.follow(source_type);
  if (!source_type.is_struct() && !source_type.is_union())
    return gen_zero(clean_type);

  member_exprt member_expr(source, attr_name, clean_type);
  return member_expr;
}

// Register instance attribute in maps
void python_converter::register_instance_attribute(
  const std::string &symbol_id,
  const std::string &attr_name,
  const std::string &var_name,
  const std::string &class_tag)
{
  // Add to regular instance attribute map
  instance_attr_map[symbol_id].insert(attr_name);

  // For 'self' parameters, also track with normalized key for cross-method access
  if (var_name == "self")
  {
    std::string normalized_key = create_normalized_self_key(class_tag);
    instance_attr_map[normalized_key].insert(attr_name);
  }
}

bool python_converter::is_instance_attribute(
  const std::string &symbol_id,
  const std::string &attr_name,
  const std::string &var_name,
  const std::string &class_tag)
{
  // Check regular per-symbol lookup
  auto it = instance_attr_map.find(symbol_id);
  if (
    it != instance_attr_map.end() &&
    it->second.find(attr_name) != it->second.end())
    return true;

  // For 'self' parameters, check normalized key for cross-method access
  if (var_name == "self")
  {
    std::string normalized_key = create_normalized_self_key(class_tag);
    auto self_it = instance_attr_map.find(normalized_key);
    if (self_it != instance_attr_map.end())
      return self_it->second.find(attr_name) != self_it->second.end();
  }

  return false;
}

void python_converter::copy_instance_attributes(
  const std::string &src_obj_id,
  const std::string &target_obj_id)
{
  auto src_attrs = instance_attr_map.find(src_obj_id);

  if (src_attrs != instance_attr_map.end())
  {
    std::set<std::string> &target_attrs = instance_attr_map[target_obj_id];
    target_attrs.insert(src_attrs->second.begin(), src_attrs->second.end());
  }
}

void python_converter::update_instance_from_self(
  const std::string &class_name,
  const std::string &func_name,
  const std::string &obj_symbol_id)
{
  symbol_id sid(current_python_file, class_name, func_name);
  sid.set_object("self");
  copy_instance_attributes(sid.to_string(), obj_symbol_id);
}

typet python_converter::infer_attr_type_from_usage(
  const std::string &class_name,
  const std::string &attr_name)
{
  const auto &module_body = (*ast_json)["body"];

  // Normalise Assign / AnnAssign into a (target, value) pair. Returns
  // nullptrs for stmts we don't handle.
  auto tgt_val = [](const nlohmann::json &stmt)
    -> std::pair<const nlohmann::json *, const nlohmann::json *> {
    if (!stmt.is_object() || !stmt.contains("value"))
      return {nullptr, nullptr};
    const std::string &k = stmt.value("_type", "");
    if (
      k == "Assign" && stmt.contains("targets") && stmt["targets"].is_array() &&
      !stmt["targets"].empty())
      return {&stmt["targets"][0], &stmt["value"]};
    if (k == "AnnAssign" && stmt.contains("target"))
      return {&stmt["target"], &stmt["value"]};
    return {nullptr, nullptr};
  };

  // Predicate: a typet produced by the helpers below is "unset" if it was
  // default-constructed (empty id) or explicitly nil. Either form means
  // the caller had no type information to offer.
  auto unset = [](const typet &ty) {
    return ty.id().as_string().empty() || ty.is_nil();
  };

  // Build a variable-to-class map from module-level assignments. Covers:
  //   <Name> = <Cls>(...)    — direct instantiation
  //   <Name> = <other Name>  — single-hop alias to a known instance
  // On conflicting class for the same variable (shadowing / reassignment to a
  // different class), permanently tombstone the name so that later writes
  // never silently reattach a wrong class — erasure alone is not enough
  // because a third assignment could re-insert the variable.
  std::unordered_map<std::string, std::string> var_to_class;
  std::unordered_set<std::string> conflicted_vars;
  auto record_var = [&](const std::string &name, const std::string &cls) {
    if (conflicted_vars.count(name))
      return;
    auto it = var_to_class.find(name);
    if (it == var_to_class.end())
      var_to_class.emplace(name, cls);
    else if (it->second != cls)
    {
      var_to_class.erase(it);
      conflicted_vars.insert(name);
    }
  };
  for (const auto &stmt : module_body)
  {
    auto [t, v] = tgt_val(stmt);
    if (
      !t || !v || !t->is_object() || t->value("_type", "") != "Name" ||
      !t->contains("id") || !v->is_object())
      continue;
    const std::string &vk = v->value("_type", "");
    const std::string var_name = (*t)["id"].get<std::string>();
    if (vk == "Call")
    {
      const auto &f = (*v)["func"];
      if (f.is_object() && f.value("_type", "") == "Name" && f.contains("id"))
        record_var(var_name, f["id"].get<std::string>());
    }
    else if (vk == "Name" && v->contains("id"))
    {
      auto alias_it = var_to_class.find(v->at("id").get<std::string>());
      if (alias_it != var_to_class.end())
        record_var(var_name, alias_it->second);
    }
  }

  // Resolve a class name to a pointer-to-struct type. Uses symbol_typet so
  // the struct is resolved lazily via ns.follow() at use time — capturing
  // sym->type directly would snapshot a possibly incomplete struct layout.
  auto cls_ptr = [&](const std::string &cls) -> typet {
    if (!json_utils::is_class(cls, *ast_json))
      return typet();
    return gen_pointer_type(symbol_typet("tag-" + cls));
  };

  // Cheap static-type inference for an RHS JSON node.
  auto infer_rhs = [&](const nlohmann::json &rhs) -> typet {
    if (!rhs.is_object())
      return typet();
    const std::string k = rhs.value("_type", "");
    if (
      k == "Call" && rhs["func"].is_object() &&
      rhs["func"].value("_type", "") == "Name" && rhs["func"].contains("id"))
      return cls_ptr(rhs["func"]["id"].get<std::string>());
    if (k == "Name" && rhs.contains("id"))
    {
      auto it = var_to_class.find(rhs["id"].get<std::string>());
      if (it != var_to_class.end())
        return cls_ptr(it->second);
    }
    return typet();
  };

  // Scan `stmts` for `<base>.<attr_name> = <rhs>` where `base` satisfies
  // `base_ok` and `rhs` yields a concrete type. Walk every hit so that
  // mutually inconsistent assignments (e.g. `n1.next = node; n1.next = other`)
  // fall back to any_type() rather than silently adopting the first type.
  auto scan =
    [&](
      const nlohmann::json &stmts,
      const std::function<bool(const std::string &)> &base_ok) -> typet {
    typet first;
    for (const auto &stmt : stmts)
    {
      auto [t, v] = tgt_val(stmt);
      if (
        !t || !v || !t->is_object() || t->value("_type", "") != "Attribute" ||
        t->value("attr", "") != attr_name || !t->contains("value") ||
        !(*t)["value"].is_object() ||
        (*t)["value"].value("_type", "") != "Name" ||
        !(*t)["value"].contains("id") ||
        !base_ok((*t)["value"]["id"].get<std::string>()))
        continue;
      typet r = infer_rhs(*v);
      if (unset(r))
        continue;
      if (unset(first))
        first = r;
      else if (first != r)
        return typet();
    }
    return first;
  };

  // Preferred: module-level `<var>.<attr> = <rhs>` where <var> is a known
  // instance of class_name (handles `n1.next = n2` after `n1 = Node(1)`).
  typet t = scan(module_body, [&](const std::string &name) {
    auto it = var_to_class.find(name);
    return it != var_to_class.end() && it->second == class_name;
  });
  if (!unset(t))
    return t;

  // Fallback: `self.<attr> = <rhs>` inside the class's own methods. Unify
  // across all methods so that two methods assigning different classes to
  // the same attribute fall back to any_type() rather than returning
  // whichever is encountered first.
  const auto &cls_node = json_utils::find_class(module_body, class_name);
  if (!cls_node.is_null() && cls_node.contains("body"))
  {
    typet unified;
    for (const auto &m : cls_node.at("body"))
    {
      if (
        !m.is_object() || m.value("_type", "") != "FunctionDef" ||
        !m.contains("body"))
        continue;
      typet r =
        scan(m["body"], [](const std::string &n) { return n == "self"; });
      if (unset(r))
        continue;
      if (unset(unified))
        unified = r;
      else if (unified != r)
        return typet();
    }
    if (!unset(unified))
      return unified;
  }

  return typet();
}

void python_converter::get_attributes_from_self(
  const nlohmann::json &func_node,
  struct_typet &clazz)
{
  const nlohmann::json &method_body =
    func_node.contains("body") ? func_node.at("body") : func_node;

  // Build a map of parameter name -> annotation from the function signature.
  // Used to recover the declared type when the body annotation is uninformative
  // (e.g., "NoneType" inferred from a None literal at a call site).
  std::unordered_map<std::string, nlohmann::json> param_annotations;
  if (
    func_node.contains("args") && func_node["args"].is_object() &&
    func_node["args"].contains("args"))
  {
    for (const auto &arg : func_node["args"]["args"])
    {
      if (
        arg.contains("arg") && arg.contains("annotation") &&
        !arg["annotation"].is_null())
        param_annotations[arg["arg"].get<std::string>()] = arg["annotation"];
    }
  }

  for (const auto &stmt : method_body)
  {
    if (
      stmt.contains("_type") && stmt["_type"] == "AnnAssign" &&
      stmt.contains("target") && stmt["target"].is_object() &&
      stmt["target"].contains("_type") &&
      stmt["target"]["_type"] == "Attribute" &&
      stmt["target"].contains("value") && stmt["target"]["value"].is_object() &&
      stmt["target"]["value"].contains("id") &&
      stmt["target"]["value"]["id"] == "self")
    {
      const std::string &attr_name = stmt["target"]["attr"];

      // Handle both simple names (id) and module-qualified names (Attribute)
      std::string annotated_type;

      if (stmt["annotation"].contains("id"))
      {
        // Simple type annotation such as self._md: Bar
        annotated_type = stmt["annotation"]["id"].get<std::string>();
      }
      else if (
        stmt["annotation"].contains("_type") &&
        stmt["annotation"]["_type"] == "Attribute")
      {
        // Module-qualified type annotation like: self._md: md.Bar
        // Extract just the class name (the attribute part)
        annotated_type = stmt["annotation"]["attr"].get<std::string>();
      }
      else if (
        stmt["annotation"].contains("_type") &&
        stmt["annotation"]["_type"] == "Subscript")
      {
        // Subscript annotation like dict[str, int] or list[int]
        typet type = get_type_from_annotation(stmt["annotation"], stmt);
        if (type.is_nil())
        {
          log_warning(
            "Skipping attribute '{}' with unsupported annotation type",
            attr_name);
          continue;
        }
        if (!clazz.has_component(attr_name))
        {
          struct_typet::componentt comp = python_frontend::build_component(
            current_class_name_, attr_name, type);
          clazz.components().push_back(comp);
        }
        continue;
      }
      else
      {
        log_warning(
          "Skipping attribute '{}' with unsupported annotation type",
          attr_name);
        continue;
      }

      typet type;
      if (annotated_type == "str")
        type = gen_pointer_type(char_type());
      else if (annotated_type == "Optional")
      {
        // The body annotation may be bare "Optional" without the inner type.
        // Try to recover the full annotation (e.g., Optional["List"]) from
        // the function parameter declaration.
        typet resolved;
        if (
          stmt.contains("value") && stmt["value"].is_object() &&
          stmt["value"].contains("id"))
        {
          const std::string param_name = stmt["value"]["id"].get<std::string>();
          auto it = param_annotations.find(param_name);
          if (it != param_annotations.end())
            resolved = get_type_from_annotation(it->second, stmt);
        }
        if (!resolved.is_nil() && !resolved.is_empty())
          type = resolved;
        else
        {
          typet base_type = get_type_from_annotation(stmt["annotation"], stmt);
          type = gen_pointer_type(base_type);
        }
      }
      else if (annotated_type == "NoneType")
      {
        // The annotator inferred NoneType from a None literal at a call site.
        // Resolve the real type from (1) the declared parameter annotation
        // for `self.x = param`, else (2) non-None assignments to this
        // attribute elsewhere in the module — the latter handles linked-list
        // / tree patterns like `self.next = None` in __init__ plus
        // `n1.next = n2` at module scope.
        typet resolved;
        if (
          stmt.contains("value") && stmt["value"].is_object() &&
          stmt["value"].contains("id"))
        {
          const std::string param_name = stmt["value"]["id"].get<std::string>();
          auto it = param_annotations.find(param_name);
          if (it != param_annotations.end())
            resolved = get_type_from_annotation(it->second, stmt);
        }
        // Default-constructed typet has an empty id; explicit error paths
        // in get_type_from_annotation may return an id_nil one instead.
        // Treat both as "unset" before falling through.
        auto unset = [](const typet &ty) {
          return ty.id().as_string().empty() || ty.is_nil();
        };
        if (unset(resolved))
          resolved = infer_attr_type_from_usage(current_class_name_, attr_name);
        type = unset(resolved) ? any_type() : resolved;
      }
      else if (annotated_type == "tuple" || annotated_type == "Tuple")
      {
        // Bare `tuple` annotation: recover element types from the assigned
        // value when it is a Tuple literal so later `obj.attr` reads carry the
        // struct shape needed for unpacking (GitHub #4515).
        if (stmt.contains("value"))
          type =
            infer_tuple_struct_from_value(stmt["value"], param_annotations);
        if (type.is_nil() || type.is_empty())
          type = type_handler_.get_typet(annotated_type);
      }
      else
        type = type_handler_.get_typet(annotated_type);

      if (!clazz.has_component(attr_name))
      {
        struct_typet::componentt comp = python_frontend::build_component(
          current_class_name_, attr_name, type);
        clazz.components().push_back(comp);
      }
    }
    else if (
      stmt.contains("_type") && stmt["_type"] == "Assign" &&
      stmt.contains("targets") && stmt["targets"].is_array() &&
      !stmt["targets"].empty() && stmt["targets"][0].is_object() &&
      stmt["targets"][0].contains("_type") &&
      stmt["targets"][0]["_type"] == "Attribute" &&
      stmt["targets"][0].contains("value") &&
      stmt["targets"][0]["value"].is_object() &&
      stmt["targets"][0]["value"].contains("id") &&
      stmt["targets"][0]["value"]["id"] == "self")
    {
      // A member is initialized with something that might be not annotated
      typet type = any_type();
      const std::string &attr_name = stmt["targets"][0]["attr"];
      if (!clazz.has_component(attr_name))
      {
        struct_typet::componentt comp = python_frontend::build_component(
          current_class_name_, attr_name, type);
        clazz.components().push_back(comp);
      }
    }
  }
}

// Process forward reference
void python_converter::process_forward_reference(
  const nlohmann::json &annotation,
  codet &target_block)
{
  if (annotation.is_null())
    return;

  std::string referenced_class;

  // Process string form of forward reference: 'Bar'
  if (
    (annotation["_type"] == "Constant" || annotation["_type"] == "Str") &&
    annotation.contains("value") && !annotation["value"].is_null())
  {
    referenced_class =
      type_utils::remove_quotes(annotation["value"].get<std::string>());
  }
  // Process direct name reference: Bar
  else if (annotation["_type"] == "Name" && annotation.contains("id"))
  {
    referenced_class = annotation["id"].get<std::string>();

    if (
      type_utils::is_builtin_type(referenced_class) ||
      type_utils::is_consensus_type(referenced_class))
      return;
  }
  else
  {
    return;
  }

  // If class is already in symbol table, skip
  std::string class_id = "tag-" + referenced_class;
  if (symbol_table_.find_symbol(class_id))
    return;

  // Find and process referenced class definition
  const auto ref_class_node =
    json_utils::find_class((*ast_json)["body"], referenced_class);

  if (!ref_class_node.empty())
  {
    std::string saved_class = current_class_name_;
    std::string saved_func = current_func_name_;
    get_class_definition(ref_class_node, target_block);
    current_class_name_ = saved_class;
    current_func_name_ = saved_func;
  }
}

void python_converter::get_class_definition(
  const nlohmann::json &class_node,
  codet &target_block)
{
  python_class_builder(*this, class_node).build(target_block);
}
