#include <python_converter.h>
#include <symbol_id.h>
#include <json_utils.h>
#include <python_class_builder.h>
#include <type_utils.h>
#include <util/expr_util.h>
#include <util/irep.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/symbol.h>

// Extracts the last identifier in a dotted name, e.g. "pkg.sub.Base" → "Base"
std::string python_class_builder::leaf(const std::string &dotted)
{
  auto p = dotted.rfind('.');
  return p == std::string::npos ? dotted : dotted.substr(p + 1);
}

/* Ensures a type symbol exists for a given class name.
 * Creates an incomplete struct type if the symbol is not yet defined. */
symbolt *python_class_builder::ensure_sym(const std::string &name)
{
  const std::string id = "tag-" + name;
  if (auto *s = conv_.symbol_table_.find_symbol(id))
    return s;

  locationt loc = conv_.get_location_from_decl(cls_);
  std::string mod = loc.get_file().as_string();

  struct_typet inc;
  inc.tag(name);
  inc.incomplete(true);
  symbolt sym = conv_.create_symbol(mod, name, id, loc, inc);
  sym.is_type = true;
  return conv_.symbol_table_.move_symbol_to_context(sym);
}

/* Handles inheritance: collects user-defined base classes and
 * merges their components (fields) into the derived struct type. */
bool python_class_builder::get_bases(struct_typet &st)
{
  bool has_ud = false;
  auto &ids = st.add("bases").get_sub();

  for (const auto &bfull : pc_.bases())
  {
    std::string base = leaf(bfull);
    if (
      type_utils::is_builtin_type(base) ||
      type_utils::is_consensus_type(base) ||
      type_utils::is_typeddict(base))
      continue;

    has_ud = true;
    auto *bsym = conv_.symbol_table_.find_symbol("tag-" + base);
    if (!bsym)
      throw std::runtime_error("Base class not found: " + base);

    ids.emplace_back(bsym->id);

    auto &bty = static_cast<struct_typet &>(bsym->type);
    for (const auto &c : bty.components())
      st.components().push_back(c);
  }
  return has_ud;
}

/* Converts methods and annotated class-level attributes (AnnAssign)
 * into ESBMC symbols. Recursively processes referenced classes. */
void python_class_builder::get_members(struct_typet &st, codet &out)
{
  std::string saved_class_name = conv_.current_class_name_;
  if (conv_.current_class_name_.empty())
  {
    // Extract class name from struct tag (e.g., "tag-int" -> "int")
    std::string class_name = st.tag().as_string();
    if (class_name.starts_with("tag-"))
      class_name = class_name.substr(4);
    conv_.current_class_name_ = class_name;
  }

  for (const auto &n : cls_.at("body"))
  {
    const std::string kind = n.value("_type", "");
    if (kind == "FunctionDef")
    {
      std::string mname = n.value("name", "");
      if (mname == "__init__")
        mname = conv_.current_class_name_;

      // For class methods, we don't want the hierarchical path
      // Just the simple method name
      std::string saved_func_name = conv_.current_func_name_;
      conv_.current_func_name_ = "";

      conv_.get_function_definition(n);

      std::string saved_func_for_lookup = conv_.current_func_name_;
      conv_.current_func_name_ = mname; // Apenas o nome simples
      symbol_id method_sid = conv_.create_symbol_id();
      conv_.current_func_name_ = saved_func_for_lookup;

      symbolt *method_symbol =
        conv_.symbol_table_.find_symbol(method_sid.to_string());

      try
      {
        exprt me = symbol_expr(*method_symbol);
        st.methods().emplace_back(me.name(), me.type());
      }
      catch (const std::exception &e)
      {
        log_error(
          "Exception creating symbol_expr for {}: {}",
          method_sid.to_string(),
          e.what());
        conv_.current_func_name_ = saved_func_name;
        continue;
      }

      conv_.current_func_name_ = saved_func_name;
    }
    else if (kind == "AnnAssign")
    {
      // class-level annotated attribute
      // Check if annotation is a simple Name node with an "id" field
      // Complex types like Dict[str, Any] are Subscript nodes without direct "id"
      if (
        n.contains("annotation") && n["annotation"].contains("id") &&
        n["annotation"]["id"].is_string())
      {
        const std::string ann = n["annotation"]["id"].get<std::string>();
        if (!conv_.symbol_table_.find_symbol("tag-" + ann))
        {
          auto ref = json_utils::find_class((*conv_.ast_json)["body"], ann);
          if (!ref.empty())
          {
            auto save = conv_.current_class_name_;
            python_class_builder(conv_, ref).build(out); // recursive conversion
            conv_.current_class_name_ = save;
          }
        }
      }
      conv_.get_var_assign(n, out);

      symbol_id sid = conv_.create_symbol_id();
      sid.set_object(n["target"]["id"].get<std::string>());
      auto *sym = conv_.symbol_table_.find_symbol(sid.to_string());
      if (!sym)
        throw std::runtime_error("Class attribute not found");
      sym->static_lifetime = true;
    }
  }
  conv_.current_class_name_ = saved_class_name;
}

/* Extracts instance attributes assigned to self (e.g., self.x = ...)
 * from within method bodies and adds them as struct fields. */
void python_class_builder::add_self_attrs(struct_typet &st)
{
  // Extract instance attributes (e.g., self.x = ...) from each method body
  for (const auto &n : cls_.at("body"))
    if (n.value("_type", "") == "FunctionDef")
      conv_.get_attributes_from_self(n.at("body"), st);
}

/* Generates a default constructor (__init__) when none is provided,
 * unless there is a user-defined base class or explicit __init__. */
void python_class_builder::gen_ctor(bool has_ud_base, struct_typet &st)
{
  const bool has_init = pc_.methods().count("__init__") > 0;
  if (has_init || has_ud_base)
    return;

  code_typet f;
  f.return_type() = none_type();

  code_typet::argumentt self;
  self.type() = gen_pointer_type(st);
  self.cmt_base_name("self");
  f.arguments().push_back(self);

  locationt loc = conv_.get_location_from_decl(cls_);
  std::string mod = loc.get_file().as_string();

  symbol_id sid;
  sid.set_filename(mod);
  sid.set_class(conv_.current_class_name_);
  sid.set_function(conv_.current_class_name_);

  symbolt ctor = conv_.create_symbol(
    mod, conv_.current_class_name_, sid.to_string(), loc, f);
  ctor.value = code_blockt();
  ctor.lvalue = true;

  conv_.symbol_table_.add(ctor);
  st.methods().emplace_back(ctor.name, ctor.type);
}

// Check if any base class is TypedDict
bool python_class_builder::is_typeddict_class() const
{
  for (const auto &bfull : pc_.bases())
  {
    std::string base = leaf(bfull);
    if (type_utils::is_typeddict(base))
      return true;
  }
  return false;
}

// Main entry point: converts a Python class node into an ESBMC struct type
// and populates the symbol table with all members and metadata.
void python_class_builder::build(codet &out)
{
  // Ensure an incomplete class symbol exists
  symbolt *sym = ensure_sym(pc_.name());
  assert(sym && sym->is_type);

  // Skip if already complete (prevents infinite recursion)
  if (!sym->type.incomplete())
    return;

  sym->type.remove(irept::a_incomplete);

  // Handle TypedDict classes: they should be treated as dict types
  // TypedDict provides type hints for dictionaries but at runtime
  // they are just regular dicts
  if (is_typeddict_class())
  {
    // Create a dict type alias for this TypedDict class
    // The dict handler provides the canonical dict struct type
    typet dict_type = conv_.get_dict_handler()->get_dict_struct_type();
    sym->type = dict_type;
    conv_.current_class_name_.clear();
    return;
  }

  // Create struct type for this class
  struct_typet st;
  conv_.current_class_name_ = pc_.name();
  st.tag(conv_.current_class_name_);

  // Collect inheritance and instance attributes
  const bool has_ud_base = get_bases(st);
  add_self_attrs(st);

  // Partial commit allows nested lookups while building members
  sym->type = st;

  // Add methods, class attributes, and default constructor
  get_members(st, out);
  gen_ctor(has_ud_base, st);

  // Finalize type and clear context
  sym->type = st;
  conv_.current_class_name_.clear();
}
