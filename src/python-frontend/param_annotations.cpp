#include <python-frontend/param_annotations.h>
#include <python-frontend/json_utils.h>

#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace
{
bool is_node(const nlohmann::json &node, const char *type)
{
  return node.is_object() && node.value("_type", "") == type;
}

/// The top-level FunctionDef named @p name in @p body, or null. Unlike
/// json_utils::try_find_function this hands back the node itself, so that a
/// call site can be probed, and a parameter rewritten, without copying it.
template <typename Json>
Json *find_function(Json &body, const std::string &name)
{
  for (auto &elem : body)
    if (is_node(elem, "FunctionDef") && elem["name"] == name)
      return &elem;
  return nullptr;
}

/// True for `list` and `list[T]` (and their typing.List spellings).
bool is_list_annotation(const nlohmann::json &annotation)
{
  const bool subscript =
    is_node(annotation, "Subscript") && annotation.contains("value");
  const nlohmann::json &name = subscript ? annotation["value"] : annotation;
  return name.is_object() && name.contains("id") &&
         (name["id"] == "list" || name["id"] == "List");
}

bool is_tuple_annotation(const nlohmann::json &annotation)
{
  if (
    !is_node(annotation, "Subscript") || !annotation.contains("slice") ||
    !annotation.contains("value"))
    return false;

  const nlohmann::json &name = annotation["value"];
  return name.contains("id") &&
         (name["id"] == "tuple" || name["id"] == "Tuple");
}

/// The element annotation of `list[T]`, or null for a bare `list`.
const nlohmann::json &list_elem_annotation(const nlohmann::json &annotation)
{
  static const nlohmann::json none;
  return is_node(annotation, "Subscript") && is_list_annotation(annotation)
           ? annotation["slice"]
           : none;
}

/// Python type name of a literal constant, or "" when this pass cannot type it.
/// A bool member is deliberately left untyped: reading one back out of a stored
/// tuple is already broken without this pass (`h = [(True, 5)]; h[0][1]`
/// reports a spurious violation), so binding it would trade a false proof for a
/// false alarm.
std::string constant_type_name(const nlohmann::json &node)
{
  if (!is_node(node, "Constant") || !node.contains("value"))
    return "";

  const nlohmann::json &value = node["value"];
  if (value.is_boolean())
    return "";
  if (value.is_number_integer())
    return "int";
  if (value.is_number_float())
    return "float";
  if (value.is_string())
    return "str";
  return "";
}

/// A `tuple[T0, ...]` annotation for a tuple literal of constants, so that the
/// parameter is typed by the very code path a hand-written annotation takes.
/// Null for a tuple this pass cannot type (empty, or a non-constant element).
nlohmann::json tuple_annotation_of(const nlohmann::json &tuple_literal)
{
  nlohmann::json elts = nlohmann::json::array();
  for (const auto &elt : tuple_literal["elts"])
  {
    const std::string type_name = constant_type_name(elt);
    if (type_name.empty())
      return nlohmann::json();
    elts.push_back({{"_type", "Name"}, {"id", type_name}});
  }

  if (elts.empty())
    return nlohmann::json();

  // A 1-tuple annotation carries the lone element node directly as its slice.
  nlohmann::json slice = elts.size() == 1
                           ? elts[0]
                           : nlohmann::json{{"_type", "Tuple"}, {"elts", elts}};

  return nlohmann::json{
    {"_type", "Subscript"},
    {"value", {{"_type", "Name"}, {"id", "tuple"}}},
    {"slice", slice}};
}

/// The `tuple[...]` annotation shared by every element of a list literal. Null
/// when the literal is empty, holds non-tuples, or is heterogeneous.
nlohmann::json tuple_annotation_of_list(const nlohmann::json &list_literal)
{
  const nlohmann::json &elts = list_literal["elts"];
  if (elts.empty() || !is_node(elts[0], "Tuple"))
    return nlohmann::json();

  const nlohmann::json annotation = tuple_annotation_of(elts[0]);
  for (const auto &elt : elts)
    if (!is_node(elt, "Tuple") || tuple_annotation_of(elt) != annotation)
      return nlohmann::json();

  return annotation;
}

/// `list[<elem>]`, keeping the location fields the annotator reads off @p
/// origin.
nlohmann::json
list_annotation_of(const nlohmann::json &elem, const nlohmann::json &origin)
{
  nlohmann::json annotation{
    {"_type", "Subscript"},
    {"value", {{"_type", "Name"}, {"id", "list"}}},
    {"slice", elem}};

  for (const char *field :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (origin.contains(field))
      annotation[field] = origin[field];

  return annotation;
}

/// True when @p annotation names a module-level alias (`_Elem = int`). A model
/// spells its element type through one to mark every parameter and return
/// sharing it as that element, so that they are retyped together with the list.
bool names_alias(const nlohmann::json &body, const nlohmann::json &annotation)
{
  if (!is_node(annotation, "Name"))
    return false;

  for (const auto &stmt : body)
    if (
      json_utils::is_name_alias(stmt) &&
      stmt["targets"][0]["id"] == annotation["id"])
      return true;
  return false;
}

/// True when two annotations spell the same type, ignoring source locations.
bool same_type(const nlohmann::json &a, const nlohmann::json &b)
{
  if (is_node(a, "Name") && is_node(b, "Name"))
    return a["id"] == b["id"];
  return a == b;
}

/// Rewrite @p def's scalar parameters and return annotated @p placeholder to
/// @p elem.
void retype_placeholder(
  nlohmann::json &def,
  const nlohmann::json &placeholder,
  const nlohmann::json &elem)
{
  const auto same = [&](const nlohmann::json &annotation) {
    return is_node(annotation, "Name") && annotation["id"] == placeholder["id"];
  };

  for (auto &arg : def["args"]["args"])
    if (arg.contains("annotation") && same(arg["annotation"]))
      arg["annotation"] = elem;
  if (def.contains("returns") && same(def["returns"]))
    def["returns"] = elem;
}

/// A call's callee name: `f(...)` or `m.f(...)`. Empty when neither.
std::string callee_name(const nlohmann::json &func)
{
  if (is_node(func, "Attribute"))
    return func["attr"].get<std::string>();
  if (is_node(func, "Name"))
    return func["id"].get<std::string>();
  return "";
}

/// Identifies one parameter: its module, its function's name, its position.
using param_key = std::tuple<size_t, std::string, size_t>;

/// The tuple element annotation inferred for a parameter, reset to none once a
/// call site disagrees with another or cannot be typed.
using bindings = std::map<param_key, std::optional<nlohmann::json>>;

class propagator
{
public:
  explicit propagator(
    const std::vector<python_param_annotations::module_ast> &modules)
    : modules_(modules)
  {
    for (const auto &module : modules_)
    {
      collect_shadowing_names(module.read->at("body"), true);
      collect_live_names(module.read->at("body"));
    }
  }

  /// One pass over every call site. Returns true if an annotation changed.
  bool run() const
  {
    bindings bound;
    for (size_t module = 0; module < modules_.size(); ++module)
      visit(modules_[module].read->at("body"), "", module, bound);
    return apply(bound);
  }

private:
  const std::vector<python_param_annotations::module_ast> &modules_;

  /// Names of functions defined anywhere below module scope (nested in another
  /// function, or a method). A call to such a name may bind to the inner
  /// definition, which this pass does not track, so it must not be attributed
  /// to a same-named top-level function.
  std::set<std::string> shadowing_names_;

  void collect_shadowing_names(const nlohmann::json &node, bool at_module_scope)
  {
    if (node.is_array())
    {
      for (const auto &elem : node)
        collect_shadowing_names(elem, at_module_scope);
      return;
    }
    if (!node.is_object())
      return;

    const bool is_function = is_node(node, "FunctionDef");
    if (is_function && !at_module_scope)
      shadowing_names_.insert(node["name"].get<std::string>());

    if (is_function || is_node(node, "ClassDef"))
      collect_shadowing_names(node["body"], false);
  }

  /// Every name that is called, or mentioned as a value. A function whose name
  /// never appears in either position is *assumed* unreachable, so the calls in
  /// its body say nothing about what its callees receive. Dispatch through a
  /// string (`getattr(m, "f")()`, `globals()["f"]()`) or through a decorator
  /// alone hides the name and is not modelled — like the rest of this
  /// monomorphic pass, such programs are out of scope.
  ///
  /// Operational models make this gate essential: heapq's `_siftup` is called
  /// by `heapify` (with the caller's tuples) and also by the unused `heappop`,
  /// whose own `heap` parameter is unbound — counting the latter would conflict
  /// the former away and leave `_siftup` mis-typed.
  ///
  /// The set is keyed by bare name across every module, which
  /// over-approximates: an unrelated variable named `f` marks a dead `f` live.
  /// That direction only forgoes a binding; it never produces a wrong one.
  std::set<std::string> live_names_;

  void collect_live_names(const nlohmann::json &node)
  {
    if (node.is_array())
    {
      for (const auto &elem : node)
        collect_live_names(elem);
      return;
    }
    if (!node.is_object())
      return;

    if (is_node(node, "Call"))
      live_names_.insert(callee_name(node["func"]));
    // A bare mention (`f = heappop`) may call it later with unknown arguments.
    if (is_node(node, "Name"))
      live_names_.insert(node["id"].get<std::string>());

    for (const auto &child : node)
      collect_live_names(child);
  }

  bool defines(size_t module, const std::string &func) const
  {
    return find_function(modules_[module].read->at("body"), func) != nullptr;
  }

  /// The module defining @p func, or none when it is unknown or ambiguous.
  std::optional<size_t>
  resolve_callee(const nlohmann::json &func, size_t caller) const
  {
    const std::string name = callee_name(func);
    if (name.empty() || shadowing_names_.count(name))
      return std::nullopt;

    // m.f(...): only the named module can define the callee.
    if (is_node(func, "Attribute"))
    {
      if (!is_node(func["value"], "Name"))
        return std::nullopt;
      for (size_t i = 0; i < modules_.size(); ++i)
        if (
          modules_[i].name == func["value"]["id"].get<std::string>() &&
          defines(i, name))
          return i;
      return std::nullopt;
    }

    if (defines(caller, name))
      return caller;

    // `from m import f`: the callee lives in exactly one other module.
    std::optional<size_t> found;
    for (size_t i = 0; i < modules_.size(); ++i)
    {
      if (!defines(i, name))
        continue;
      if (found)
        return std::nullopt;
      found = i;
    }
    return found;
  }

  /// The tuple element annotation of a call argument, when it is a list of
  /// uniform tuples. Null otherwise.
  nlohmann::json elem_annotation_of(
    const nlohmann::json &arg,
    const std::string &caller_func,
    size_t caller) const
  {
    if (is_node(arg, "List"))
      return tuple_annotation_of_list(arg);

    if (!is_node(arg, "Name"))
      return nlohmann::json();

    const nlohmann::json decl = json_utils::find_var_decl(
      arg["id"].get<std::string>(), caller_func, *modules_[caller].read);
    if (decl.is_null())
      return nlohmann::json();

    // A parameter, or a `list[tuple[...]]`-annotated variable, names its
    // element type directly. An earlier round may have written that annotation.
    if (decl.contains("annotation"))
    {
      const nlohmann::json &elem = list_elem_annotation(decl["annotation"]);
      if (is_tuple_annotation(elem))
        return elem;
    }

    if (decl.contains("value") && is_node(decl["value"], "List"))
      return decl["value"]["elts"].empty()
               ? stored_elem_annotation(arg["id"].get<std::string>(), caller)
               : tuple_annotation_of_list(decl["value"]);

    return nlohmann::json();
  }

  /// The tuple annotation of every item stored into the list variable @p var,
  /// declared empty in module @p module. Items arrive through `var.append(x)`
  /// or `f(var, x)` whose callee annotates the list `list[_Elem]` and the item
  /// `_Elem` (`heappush(h, (2, 20))`). Null when there is no such item, items
  /// disagree, or the list may be filled some other way. Every scope of the
  /// module is searched, which only ever forgoes a binding.
  nlohmann::json
  stored_elem_annotation(const std::string &var, size_t module) const
  {
    std::vector<nlohmann::json> items;
    collect_stored_items(modules_[module].read->at("body"), var, module, items);

    if (items.empty())
      return nlohmann::json();
    for (const auto &item : items)
      if (item.is_null() || item != items[0])
        return nlohmann::json();
    return items[0];
  }

  /// Push onto @p items the tuple annotation of each item stored into @p var,
  /// or null for a store this pass cannot type.
  void collect_stored_items(
    const nlohmann::json &node,
    const std::string &var,
    size_t module,
    std::vector<nlohmann::json> &items) const
  {
    if (node.is_array())
    {
      for (const auto &elem : node)
        collect_stored_items(elem, var, module, items);
      return;
    }
    if (!node.is_object())
      return;

    if (is_untyped_store(node, var))
      items.push_back(nlohmann::json());
    else if (is_node(node, "Call"))
    {
      const std::optional<nlohmann::json> item = stored_item(node, var, module);
      if (item)
        items.push_back(
          is_node(*item, "Tuple") ? tuple_annotation_of(*item)
                                  : nlohmann::json());
    }

    for (const auto &child : node)
      collect_stored_items(child, var, module, items);
  }

  /// The item a call stores into @p var, or none when it stores nothing.
  std::optional<nlohmann::json> stored_item(
    const nlohmann::json &call,
    const std::string &var,
    size_t module) const
  {
    const nlohmann::json &func = call["func"];
    const nlohmann::json &args = call["args"];
    if (is_node(func, "Attribute") && names_var(func["value"], var))
    {
      const std::string &method = func["attr"].get<std::string>();
      if (method == "append" && args.size() == 1)
        return args[0];
      if (method == "extend" || method == "insert" || method == "append")
        return nlohmann::json();
      return std::nullopt;
    }

    for (size_t i = 0; i < args.size(); ++i)
      if (names_var(args[i], var))
        return item_param_arg(call, i, module);
    return std::nullopt;
  }

  /// The argument a call passes for the `_Elem` item parameter of a callee
  /// whose parameter @p list_pos is `list[_Elem]`. None for any other callee.
  std::optional<nlohmann::json> item_param_arg(
    const nlohmann::json &call,
    size_t list_pos,
    size_t caller) const
  {
    const std::optional<size_t> callee = resolve_callee(call["func"], caller);
    if (!callee)
      return std::nullopt;

    const nlohmann::json &body = modules_[*callee].read->at("body");
    const nlohmann::json &params =
      find_function(body, callee_name(call["func"]))->at("args").at("args");
    if (list_pos >= params.size())
      return std::nullopt;

    const nlohmann::json placeholder = list_elem_annotation(
      params[list_pos].value("annotation", nlohmann::json()));
    if (!names_alias(body, placeholder) && !is_tuple_annotation(placeholder))
      return std::nullopt;

    for (size_t i = 0; i < params.size() && i < call["args"].size(); ++i)
      if (same_type(
            params[i].value("annotation", nlohmann::json()), placeholder))
        return call["args"][i];
    return std::nullopt;
  }

  static bool names_var(const nlohmann::json &node, const std::string &var)
  {
    return is_node(node, "Name") && node["id"] == var;
  }

  /// True for `var += ...`, `var[i] = ...` and a rebinding of @p var to
  /// anything but `[]`, which may store items this pass does not see.
  static bool
  is_untyped_store(const nlohmann::json &node, const std::string &var)
  {
    if (is_node(node, "AugAssign"))
      return names_var(node["target"], var);

    nlohmann::json targets;
    if (is_node(node, "Assign"))
      targets = node["targets"];
    else if (is_node(node, "AnnAssign"))
      targets = nlohmann::json::array({node["target"]});
    else
      return false;

    const nlohmann::json &value = node.value("value", nlohmann::json());
    const bool empty_list = is_node(value, "List") && value["elts"].empty();
    for (const auto &target : targets)
      if (
        (is_node(target, "Subscript") && names_var(target["value"], var)) ||
        (names_var(target, var) && !empty_list))
        return true;
    return false;
  }

  void visit_call(
    const nlohmann::json &call,
    const std::string &caller_func,
    size_t caller,
    bindings &bound) const
  {
    const std::optional<size_t> callee = resolve_callee(call["func"], caller);
    if (!callee)
      return;

    const std::string name = callee_name(call["func"]);
    const nlohmann::json &params =
      find_function(modules_[*callee].read->at("body"), name)
        ->at("args")
        .at("args");

    for (size_t i = 0; i < params.size() && i < call["args"].size(); ++i)
    {
      if (!is_list_annotation(params[i].value("annotation", nlohmann::json())))
        continue;

      const nlohmann::json elem =
        elem_annotation_of(call["args"][i], caller_func, caller);
      std::optional<nlohmann::json> inferred;
      if (!elem.is_null())
        inferred = elem;

      auto [entry, inserted] =
        bound.emplace(param_key{*callee, name, i}, inferred);
      if (!inserted && entry->second != inferred)
        entry->second = std::nullopt; // call sites disagree
    }
  }

  /// Walk every expression, tracking the enclosing function so that a Name
  /// argument resolves against it. Class bodies are skipped: methods are not
  /// retyped by this pass.
  void visit(
    const nlohmann::json &node,
    const std::string &func,
    size_t module,
    bindings &bound) const
  {
    if (node.is_array())
    {
      for (const auto &elem : node)
        visit(elem, func, module, bound);
      return;
    }
    if (!node.is_object() || is_node(node, "ClassDef"))
      return;

    if (is_node(node, "FunctionDef"))
    {
      const std::string &name = node["name"].get<std::string>();
      if (live_names_.count(name))
        visit(
          node["body"],
          func.empty() ? name : func + "@F@" + name,
          module,
          bound);
      return;
    }

    if (is_node(node, "Call"))
      visit_call(node, func, module, bound);

    for (const auto &child : node)
      visit(child, func, module, bound);
  }

  bool apply(const bindings &bound) const
  {
    bool changed = false;
    for (const auto &[key, elem] : bound)
    {
      const auto &[module, func, param] = key;
      if (!elem || modules_[module].write == nullptr)
        continue;

      nlohmann::json &body = modules_[module].write->at("body");
      nlohmann::json &def = *find_function(body, func);
      nlohmann::json &annotation = def["args"]["args"][param]["annotation"];

      // Write once. A binding is only produced in the round where every call
      // site of the parameter is typable, and each site's type then comes from
      // a literal or from an annotation this pass has already fixed, so it can
      // never improve later. Refusing to rewrite an element type that is
      // already a tuple makes each changing round strictly reduce the number of
      // non-tuple list parameters, which is what bounds the loop below.
      if (is_tuple_annotation(list_elem_annotation(annotation)))
        continue;

      const nlohmann::json &placeholder = list_elem_annotation(annotation);
      if (names_alias(body, placeholder))
        retype_placeholder(def, placeholder, *elem);

      annotation = list_annotation_of(*elem, annotation);
      changed = true;
    }
    return changed;
  }
};
} // namespace

namespace python_param_annotations
{
void propagate_tuple_list_params(const std::vector<module_ast> &modules)
{
  const propagator pass(modules);

  // Iterate: an argument may itself be a parameter of the enclosing function,
  // whose annotation is only corrected once its own callers have been seen
  // (`heapify(heap)` -> `_siftup(heap, pos)`). Every changing round retypes at
  // least one list parameter to a tuple and never back (see apply()), so the
  // number of list parameters bounds the rounds and the loop terminates. A
  // fixed cap would instead leave a deep enough call chain mis-typed, i.e.
  // silently unsound.
  while (pass.run())
    ;
}
} // namespace python_param_annotations
