#include <util/location.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/symbol.h>

symbolt::symbolt()
{
  clear();
}

void symbolt::clear()
{
  location.make_nil();
  lvalue = static_lifetime = file_local = is_extern = is_type = is_parameter =
    is_macro = is_thread_local = false;
  is_set = false;
  python_annotation_types.clear();
  id = module = name = mode = "";
  // Both representations on each side reset to nil consistently; no
  // migration is invoked. Reading either side returns a nil/default form.
  type_ = type2tc();
  legacy_type_cache_ = typet();
  legacy_type_valid_ = true;
  type2_valid_ = true;
  value_ = expr2tc();
  legacy_value_cache_.make_nil();
  legacy_value_valid_ = true;
  value2_valid_ = true;
}

// Type setters. Each setter writes one side and invalidates the other;
// the read side derives lazily via migrate_type_back / migrate_type on
// first access. The lazy split avoids forward-migrating typets whose
// sub-expressions (e.g. an array size built from a legacy binary_exprt
// with no type set) would not survive the recursive descent.
void symbolt::set_type(const typet &t)
{
  legacy_type_cache_ = t;
  legacy_type_valid_ = true;
  type2_valid_ = false;
}

void symbolt::set_type(typet &&t)
{
  legacy_type_cache_ = std::move(t);
  legacy_type_valid_ = true;
  type2_valid_ = false;
}

void symbolt::set_type(const type2tc &t)
{
  type_ = t;
  type2_valid_ = true;
  legacy_type_valid_ = false;
}

// Value setters. Mirror of the type setters: each writes one side and
// invalidates the other; the read side derives lazily via migrate_expr /
// migrate_expr_back. The back direction covers every expr2t kind a symbol
// value may hold, including code_block2t for function bodies.
void symbolt::set_value(const exprt &v)
{
  legacy_value_cache_ = v;
  legacy_value_valid_ = true;
  value2_valid_ = false;
}

void symbolt::set_value(exprt &&v)
{
  legacy_value_cache_ = std::move(v);
  legacy_value_valid_ = true;
  value2_valid_ = false;
}

void symbolt::set_value(const expr2tc &v)
{
  value_ = v;
  value2_valid_ = true;
  legacy_value_valid_ = false;
}

const typet &symbolt::get_type() const
{
  if (!legacy_type_valid_)
  {
    // A nil IREP2 source must not be fed to migrate_type_back (it derefs
    // the held pointer). Return a default `typet` instead; this matches
    // the pre-S5a behaviour of a freshly-cleared symbolt whose legacy
    // `type` field was default-constructed.
    legacy_type_cache_ =
      is_nil_type(type_) ? typet() : migrate_type_back(type_);
    legacy_type_valid_ = true;
  }
  return legacy_type_cache_;
}

const type2tc &symbolt::get_type2() const
{
  if (!type2_valid_)
  {
    // Symmetric to get_type(): an empty-id legacy typet is the only safe
    // input migrate_type cannot consume, so it maps to a nil IREP2 form.
    type_ = legacy_type_cache_.id().empty() ? type2tc()
                                            : migrate_type(legacy_type_cache_);
    type2_valid_ = true;
  }
  return type_;
}

const exprt &symbolt::get_value() const
{
  if (!legacy_value_valid_)
  {
    // Mirror of get_type(): a nil IREP2 value must not be fed to
    // migrate_expr_back (it derefs the held pointer). Return a nil exprt
    // instead -- the shape a freshly-cleared symbolt has on the legacy
    // side. For non-nil values, migrate_expr_back covers every expr2t
    // kind a symbol value may hold, including function bodies.
    if (is_nil_expr(value_))
      legacy_value_cache_.make_nil();
    else
      legacy_value_cache_ = migrate_expr_back(value_);
    legacy_value_valid_ = true;
  }
  return legacy_value_cache_;
}

const expr2tc &symbolt::get_value2() const
{
  if (!value2_valid_)
  {
    // Symmetric to get_type2(). A nil or empty-id legacy exprt maps to a
    // nil IREP2 value -- the only inputs migrate_expr cannot consume.
    if (legacy_value_cache_.is_nil() || legacy_value_cache_.id().empty())
      value_ = expr2tc();
    else
      migrate_expr(legacy_value_cache_, value_);
    value2_valid_ = true;
  }
  return value_;
}

void symbolt::swap(symbolt &b)
{
#define SYM_SWAP1(x) x.swap(b.x)

  SYM_SWAP1(id);
  SYM_SWAP1(module);
  SYM_SWAP1(name);
  SYM_SWAP1(mode);
  SYM_SWAP1(location);
  SYM_SWAP1(python_annotation_types);
  SYM_SWAP1(legacy_type_cache_);
  SYM_SWAP1(legacy_value_cache_);

#define SYM_SWAP2(x) std::swap(x, b.x)

  SYM_SWAP2(is_type);
  SYM_SWAP2(is_macro);
  SYM_SWAP2(is_parameter);
  SYM_SWAP2(lvalue);
  SYM_SWAP2(static_lifetime);
  SYM_SWAP2(file_local);
  SYM_SWAP2(is_extern);
  SYM_SWAP2(is_thread_local);
  SYM_SWAP2(is_set);
  SYM_SWAP2(type_);
  SYM_SWAP2(legacy_type_valid_);
  SYM_SWAP2(type2_valid_);
  SYM_SWAP2(value_);
  SYM_SWAP2(legacy_value_valid_);
  SYM_SWAP2(value2_valid_);
}

void symbolt::dump() const
{
  std::ostringstream oss;
  show(oss);
  log_status("{}", oss.str());
}

void symbolt::show(std::ostream &out) const
{
  out << "Symbol......: " << id << "\n";
  out << "Base name...: " << name << "\n";
  out << "Module......: " << module << "\n";
  out << "Mode........: " << mode << " (" << mode << ")"
      << "\n";
  // Read the type/value through the accessors: the legacy fields are now
  // derived caches, so direct references would expose stale data.
  const typet &t = get_type();
  if (t.is_not_nil())
    out << "Type........: " << t.pretty(4) << "\n";
  const exprt &v = get_value();
  if (v.is_not_nil())
    out << "Value.......: " << v.pretty(4) << "\n";

  out << "Flags.......:";

  if (lvalue)
    out << " lvalue";
  if (static_lifetime)
    out << " static_lifetime";
  if (file_local)
    out << " file_local";
  if (is_type)
    out << " type";
  if (is_extern)
    out << " extern";
  if (is_macro)
    out << " macro";
  if (is_thread_local)
    out << " is_thread_local";

  out << "\n";
  out << "Location....: " << location << "\n";

  out << "\n";
}

std::ostream &operator<<(std::ostream &out, const symbolt &symbol)
{
  symbol.show(out);
  return out;
}

void symbolt::to_irep(irept &dest) const
{
  dest.clear();
  // Derive the legacy `typet` and `exprt` from the IREP2 source via
  // get_type() / get_value() and serialize as before -- same on-disk
  // format, no goto-binary change.
  dest.type() = get_type();
  dest.symvalue(get_value());
  dest.location(location);
  dest.name(id);
  dest.module(module);
  dest.base_name(name);
  dest.mode(mode);

  if (is_type)
    dest.is_type(true);
  if (is_macro)
    dest.is_macro(true);
  if (is_parameter)
    dest.is_parameter(true);
  if (lvalue)
    dest.lvalue(true);
  if (static_lifetime)
    dest.static_lifetime(true);
  if (file_local)
    dest.file_local(true);
  if (is_extern)
    dest.is_extern(true);
  if (is_thread_local)
    dest.is_thread_local(true);

  if (!python_annotation_types.empty())
  {
    irept &annotations = dest.add("python_annotations");
    auto &sub = annotations.get_sub();
    sub.reserve(python_annotation_types.size());
    for (const auto &type : python_annotation_types)
      sub.push_back(type);
  }
}

void symbolt::from_irep(const irept &src)
{
  // Bridge the on-disk legacy form into the legacy caches; the IREP2
  // representations are derived lazily on the next get_type2() /
  // get_value2() call (matching the setter semantics -- forward migration
  // is never eager).
  legacy_type_cache_ = src.type();
  legacy_type_valid_ = true;
  type2_valid_ = false;

  legacy_value_cache_ = static_cast<const exprt &>(src.symvalue());
  legacy_value_valid_ = true;
  value2_valid_ = false;

  location = static_cast<const locationt &>(src.location());

  id = src.name();
  module = src.module();
  name = src.base_name();
  mode = src.mode();

  is_type = src.is_type();
  is_macro = src.is_macro();
  is_parameter = src.is_parameter();
  lvalue = src.lvalue();
  static_lifetime = src.static_lifetime();
  file_local = src.file_local();
  is_extern = src.is_extern();
  is_thread_local = src.is_thread_local();
  is_set = false;
  python_annotation_types.clear();
  const irept &annotations = src.find("python_annotations");
  if (!annotations.is_nil())
  {
    for (const auto &type : annotations.get_sub())
      python_annotation_types.emplace_back(static_cast<const typet &>(type));
  }
}

irep_idt symbolt::get_function_name() const
{
  irep_idt func_name = location.get_function();
  if (!func_name.empty())
    return func_name;

  const std::string &symbol_id = id.as_string();

  // Find the position of "F@"
  size_t posF = symbol_id.find("F@");

  if (posF == std::string::npos)
    return ""; // Return an empty string if "F@" is not found

  posF += 2; // Advance beyond "F@"

  // Find the position of the last '@'
  size_t posLastAt = symbol_id.rfind('@');

  // Check if there is an '@' after the function name (e.g.: c:string.c@1290@F@strcmp@c1)
  if (posLastAt != std::string::npos && posLastAt > posF)
    return symbol_id.substr(
      posF,
      posLastAt - posF); // Extract the content between "F@" and the last '@'

  return symbol_id.substr(
    posF); // If there is no '@' after the function name (e.g: c:string.c@1290@F@strcmp), return from "F@"
}
