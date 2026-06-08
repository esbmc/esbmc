#include <cassert>
#include <cstdlib>
#include <util/i2string.h>
#include <util/irep.h>
#include <util/message.h>

irept nil_rep_storage;

#ifdef SHARING
const irept::dt empty_d;
#endif

void irept::dump() const
{
  log_status("{}", pretty(0));
}

const irept &get_nil_irep()
{
  if (nil_rep_storage.id().empty()) // initialized?
    nil_rep_storage.id("nil");
  return nil_rep_storage;
}

#ifdef SHARING
irept::irept(const irep_idt &_id) : data(new dt)
{
  id(_id);
}
#else
irept::irept(const irep_idt &_id)
{
  id(_id);
}
#endif

#ifdef SHARING
void irept::detatch()
{
  if (data == nullptr)
  {
    data = new dt;
    assert(data->ref_count == 1);
    return;
  }

  dt *const old_data = data;
  old_data->dt_mutex.lock();
  if (old_data->ref_count == 1)
  {
    old_data->dt_mutex.unlock();
    return;
  }
  dt *new_data = new dt(*old_data);
  this->data = new_data;

  data->ref_count = 1;

  old_data->dt_mutex.unlock();
  remove_ref(old_data);
}
#endif

#ifdef SHARING
const irept::dt &irept::read() const
{
  if (data == nullptr)
    return empty_d;

  return *data;
}
#endif

#ifdef SHARING
void irept::remove_ref(dt *old_data)
{
  bool should_delete = false;

  if (old_data == nullptr)
    return;

  {
    // Lock the data block we are about to modify.
    std::lock_guard<std::mutex> lock(old_data->dt_mutex);
    assert(old_data->ref_count > 0);
    old_data->ref_count--;
    if (old_data->ref_count == 0)
      should_delete = true;
  }

  // Delete the data after releasing the lock.
  if (should_delete)
  {
    delete old_data;
  }
}
#endif

void irept::clear()
{
#ifdef SHARING
  remove_ref(data);
  data = nullptr;
#else
  data.clear();
#endif
}

void irept::move_to_named_sub(const irep_idt &name, irept &irep)
{
#ifdef SHARING
  detatch();
#endif
  add(name).swap(irep);
  irep.clear();
}

void irept::move_to_sub(irept &irep)
{
#ifdef SHARING
  detatch();
#endif
  get_sub().push_back(get_nil_irep());
  get_sub().back().swap(irep);
}

const irep_idt &irept::get(const irep_idt &name) const
{
  const named_subt &s = is_comment(name) ? get_comments() : get_named_sub();

  named_subt::const_iterator it = s.find(name);

  if (it == s.end())
  {
    const static irep_idt empty;
    return empty;
  }

  return it->second.id();
}

bool irept::get_bool(const irep_idt &name) const
{
  return atoi(get(name).c_str());
}

void irept::set(const irep_idt &name, const long value)
{
  add(name).id(i2string((int)value));
}

void irept::remove(const irep_idt &name)
{
  named_subt &s = is_comment(name) ? get_comments() : get_named_sub();

  named_subt::iterator it = s.find(name);

  if (it != s.end())
    s.erase(it);
}

void irept::set(const irep_idt &name, const irept &irep)
{
  add(name) = irep;
}

const irept &irept::find(const irep_idt &name) const
{
  const named_subt &s = is_comment(name) ? get_comments() : get_named_sub();

  named_subt::const_iterator it = s.find(name);

  if (it == s.end())
    return get_nil_irep();

  return it->second;
}

irept &irept::add(const irep_idt &name)
{
  named_subt &s = is_comment(name) ? get_comments() : get_named_sub();

  return s[name];
}

bool operator==(const irept &i1, const irept &i2)
{
#ifdef SHARING
  if (i1.data == i2.data)
    return true;
#endif

  if (i1.id() != i2.id())
    return false;

  if (i1.get_sub() != i2.get_sub())
    return false; // recursive call

  if (i1.get_named_sub() != i2.get_named_sub())
    return false; // recursive call

  // comments are NOT checked

  return true;
}

bool full_eq(const irept &i1, const irept &i2)
{
#ifdef SHARING
  if (i1.data == i2.data)
    return true;
#endif

  if (i1.id() != i2.id())
    return false;

  const irept::subt &i1_sub = i1.get_sub();
  const irept::subt &i2_sub = i2.get_sub();
  const irept::named_subt &i1_named_sub = i1.get_named_sub();
  const irept::named_subt &i2_named_sub = i2.get_named_sub();
  const irept::named_subt &i1_comments = i1.get_comments();
  const irept::named_subt &i2_comments = i2.get_comments();

  if (i1_sub.size() != i2_sub.size())
    return false;
  if (i1_named_sub.size() != i2_named_sub.size())
    return false;
  if (i1_comments.size() != i2_comments.size())
    return false;

  for (unsigned i = 0; i < i1_sub.size(); i++)
    if (!full_eq(i1_sub[i], i2_sub[i]))
      return false;

  {
    irept::named_subt::const_iterator i1_it = i1_named_sub.begin();
    irept::named_subt::const_iterator i2_it = i2_named_sub.begin();

    for (; i1_it != i1_named_sub.end(); ++i1_it, ++i2_it)
      if (
        i1_it->first != i2_it->first || !full_eq(i1_it->second, i2_it->second))
        return false;
  }

  {
    irept::named_subt::const_iterator i1_it = i1_comments.begin();
    irept::named_subt::const_iterator i2_it = i2_comments.begin();

    for (; i1_it != i1_comments.end(); ++i1_it, ++i2_it)
      if (
        i1_it->first != i2_it->first || !full_eq(i1_it->second, i2_it->second))
        return false;
  }

  return true;
}

std::string irept::to_string() const
{
  return pretty(0);
}

std::ostream &operator<<(std::ostream &out, const irept &irep)
{
  out << irep.to_string();
  return out;
}

bool ordering(const irept &X, const irept &Y)
{
  return X.compare(Y) < 0;

#if 0
  if(X.data<Y.data) return true;
  if(Y.data<X.data) return false;

  if(X.sub.size()<Y.sub.size()) return true;
  if(Y.sub.size()<X.sub.size()) return false;

  {
    irept::subt::const_iterator it1, it2;

    for(it1=X.sub.begin(),
        it2=Y.sub.begin();
        it1!=X.sub.end() && it2!=Y.sub.end();
        it1++,
        it2++)
    {
      if(ordering(*it1, *it2)) return true;
      if(ordering(*it2, *it1)) return false;
    }

    assert(it1==X.sub.end() && it2==Y.sub.end());
  }

  if(X.named_sub.size()<Y.named_sub.size()) return true;
  if(Y.named_sub.size()<X.named_sub.size()) return false;

  {
    irept::named_subt::const_iterator it1, it2;

    for(it1=X.named_sub.begin(),
        it2=Y.named_sub.begin();
        it1!=X.named_sub.end() && it2!=Y.named_sub.end();
        it1++,
        it2++)
    {
      if(it1->first<it2->first) return true;
      if(it2->first<it1->first) return false;

      if(ordering(it1->second, it2->second)) return true;
      if(ordering(it2->second, it1->second)) return false;
    }

    assert(it1==X.named_sub.end() && it2==Y.named_sub.end());
  }

  return false;
#endif
}

int irept::compare(const irept &i) const
{
  int r;

  r = id().compare(i.id());
  if (r != 0)
    return r;

  if (get_sub().size() < i.get_sub().size())
    return -1;
  if (get_sub().size() > i.get_sub().size())
    return 1;

  {
    irept::subt::const_iterator it1, it2;

    for (it1 = get_sub().begin(), it2 = i.get_sub().begin();
         it1 != get_sub().end() && it2 != i.get_sub().end();
         ++it1, ++it2)
    {
      r = it1->compare(*it2);
      if (r != 0)
        return r;
    }

    assert(it1 == get_sub().end() && it2 == i.get_sub().end());
  }

  if (get_named_sub().size() < i.get_named_sub().size())
    return -1;
  if (get_named_sub().size() > i.get_named_sub().size())
    return 1;

  {
    irept::named_subt::const_iterator it1, it2;

    for (it1 = get_named_sub().begin(), it2 = i.get_named_sub().begin();
         it1 != get_named_sub().end() && it2 != i.get_named_sub().end();
         ++it1, ++it2)
    {
      r = it1->first.compare(it2->first);
      if (r != 0)
        return r;

      r = it1->second.compare(it2->second);
      if (r != 0)
        return r;
    }

    assert(it1 == get_named_sub().end() && it2 == i.get_named_sub().end());
  }

  // equal
  return 0;
}

bool operator<(const irept &X, const irept &Y)
{
  return ordering(X, Y);
}

size_t irept::hash() const
{
  size_t result = std::hash<std::string>{}(id().as_string());

  forall_irep (it, get_sub())
    result = result ^ it->hash();

  forall_named_irep (it, get_named_sub())
  {
    result = result ^ std::hash<std::string>{}(it->first.as_string());
    result = result ^ it->second.hash();
  }

  return result;
}

size_t irept::full_hash() const
{
  size_t result = std::hash<std::string>{}(id().as_string());

  forall_irep (it, get_sub())
    result = result ^ it->full_hash();

  forall_named_irep (it, get_named_sub())
  {
    result = result ^ std::hash<std::string>{}(it->first.as_string());
    result = result ^ it->second.full_hash();
  }

  forall_named_irep (it, get_comments())
  {
    result = result ^ std::hash<std::string>{}(it->first.as_string());
    result = result ^ it->second.full_hash();
  }

  return result;
}

static void indent_str(std::string &s, unsigned indent)
{
  for (unsigned i = 0; i < indent; i++)
    s += ' ';
}

std::string irept::pretty(unsigned indent) const
{
  std::string result;

  if (id() != "")
  {
    result += id_string();
    indent += 2;
  }

  forall_named_irep (it, get_named_sub())
  {
    result += "\n";
    indent_str(result, indent);

    result += "* ";
    result += it->first.as_string();
    result += ": ";

    result += it->second.pretty(indent + 2);
  }

  forall_named_irep (it, get_comments())
  {
    result += "\n";
    indent_str(result, indent);

    result += "* ";
    result += it->first.as_string();
    result += ": ";

    result += it->second.pretty(indent + 2);
  }

  unsigned count = 0;

  forall_irep (it, get_sub())
  {
    result += "\n";
    indent_str(result, indent);

    result += i2string(count++);
    result += ": ";

    result += it->pretty(indent + 2);
  }

  return result;
}

const irep_idt irept::a_width = irep_idt("width");
const irep_idt irept::a_name = irep_idt("name");
const irep_idt irept::a_statement = irep_idt("statement");
const irep_idt irept::a_identifier = irep_idt("identifier");
const irep_idt irept::a_comp_name = irep_idt("component_name");
const irep_idt irept::a_tag = irep_idt("tag");
const irep_idt irept::a_from = irep_idt("from");
const irep_idt irept::a_file = irep_idt("file");
const irep_idt irept::a_line = irep_idt("line");
const irep_idt irept::a_function = irep_idt("function");
const irep_idt irept::a_column = irep_idt("column");
const irep_idt irept::a_access = irep_idt("access");
const irep_idt irept::a_destination = irep_idt("destination");
const irep_idt irept::a_base_name = irep_idt("base_name");
const irep_idt irept::a_comment = irep_idt("comment");
const irep_idt irept::a_event = irep_idt("event");
const irep_idt irept::a_literal = irep_idt("literal");
const irep_idt irept::a_loopid = irep_idt("loop-id");
const irep_idt irept::a_mode = irep_idt("mode");
const irep_idt irept::a_module = irep_idt("module");
const irep_idt irept::a_pretty_name = irep_idt("pretty_name");
const irep_idt irept::a_property = irep_idt("property");
const irep_idt irept::a_size = irep_idt("size");
const irep_idt irept::a_integer_bits = irep_idt("integer_bits");
const irep_idt irept::a_to = irep_idt("to");
const irep_idt irept::a_failed_symbol = irep_idt("#failed_symbol");
const irep_idt irept::a_dynamic = irep_idt("#dynamic");
const irep_idt irept::a_cmt_base_name = irep_idt("#base_name");
const irep_idt irept::a_id_class = irep_idt("#id_class");
const irep_idt irept::a_cmt_identifier = irep_idt("#identifier");
const irep_idt irept::a_cformat = irep_idt("#cformat");
const irep_idt irept::a_cmt_width = irep_idt("#width");
const irep_idt irept::a_axiom = irep_idt("axiom");
const irep_idt irept::a_cmt_constant = irep_idt("#constant");
const irep_idt irept::a_default = irep_idt("default");
const irep_idt irept::a_ellipsis = irep_idt("ellipsis");
const irep_idt irept::a_explicit = irep_idt("explicit");
const irep_idt irept::a_file_local = irep_idt("file_local");
const irep_idt irept::a_hex_or_oct = irep_idt("#hex_or_oct");
const irep_idt irept::a_hide = irep_idt("#hide");
const irep_idt irept::a_implicit = irep_idt("#implicit");
const irep_idt irept::a_incomplete = irep_idt("#incomplete");
const irep_idt irept::a_initialization = irep_idt("initialization");
const irep_idt irept::a_inlined = irep_idt("#inlined");
const irep_idt irept::a_invalid_object = irep_idt("#invalid_object");
const irep_idt irept::a_is_parameter = irep_idt("is_parameter");
const irep_idt irept::a_is_expression = irep_idt("#is_expression");
const irep_idt irept::a_is_extern = irep_idt("is_extern");
const irep_idt irept::a_is_macro = irep_idt("is_macro");
const irep_idt irept::a_is_thread_local = irep_idt("is_thread_local");
const irep_idt irept::a_is_type = irep_idt("is_type");
const irep_idt irept::a_lvalue = irep_idt("lvalue");
const irep_idt irept::a_reference = irep_idt("#reference");
const irep_idt irept::a_restricted = irep_idt("#restricted");
const irep_idt irept::a_static_lifetime = irep_idt("static_lifetime");
const irep_idt irept::a_theorem = irep_idt("theorem");
const irep_idt irept::a_cmt_unsigned = irep_idt("#unsigned");
const irep_idt irept::a_user_provided = irep_idt("user-provided");
const irep_idt irept::a_cmt_volatile = irep_idt("#volatile");
const irep_idt irept::a_zero_initializer = irep_idt("#zero_initializer");
const irep_idt irept::a_flavor = irep_idt("flavor");
const irep_idt irept::a_cmt_active = irep_idt("#active");
const irep_idt irept::a_code = irep_idt("code");
const irep_idt irept::a_component = irep_idt("component");
const irep_idt irept::a_c_sizeof_type = irep_idt("#c_sizeof_type");
const irep_idt irept::a_end_location = irep_idt("#end_location");
const irep_idt irept::a_guard = irep_idt("guard");
const irep_idt irept::a_label = irep_idt("label");
const irep_idt irept::a_lhs = irep_idt("lhs");
const irep_idt irept::a_location = irep_idt("location");
const irep_idt irept::a_object_type = irep_idt("object_type");
const irep_idt irept::a_cmt_size = irep_idt("#size");
const irep_idt irept::a_cmt_type = irep_idt("#type");
const irep_idt irept::a_type_id = irep_idt("typeid");
const irep_idt irept::a_derived_this_arg = irep_idt("#derived_this_arg");
const irep_idt irept::a_base_ctor_derived = irep_idt("#base_ctor_derived");
const irep_idt irept::a_need_vptr_init = irep_idt("#need_vptr_init");

const irep_idt irept::s_type = irep_idt("type");
const irep_idt irept::s_arguments = irep_idt("arguments");
const irep_idt irept::s_components = irep_idt("components");
const irep_idt irept::s_return_type = irep_idt("return_type");
const irep_idt irept::s_body = irep_idt("body");
const irep_idt irept::s_member = irep_idt("member");
const irep_idt irept::s_labels = irep_idt("labels");
const irep_idt irept::s_bv = irep_idt("bv");
const irep_idt irept::s_targets = irep_idt("targets");
const irep_idt irept::s_variables = irep_idt("variables");
const irep_idt irept::s_initializer = irep_idt("initializer");
const irep_idt irept::s_declaration_type = irep_idt("declaration_type");
const irep_idt irept::s_decl_value = irep_idt("decl_value");
const irep_idt irept::s_symvalue = irep_idt("symvalue");
const irep_idt irept::s_cmt_location = irep_idt("#location");
const irep_idt irept::s_decl_ident = irep_idt("decl_ident");
const irep_idt irept::s_elements = irep_idt("elements");
const irep_idt irept::s_offsetof_type = irep_idt("offsetof_type");

const irep_idt irept::id_address_of = irep_idt("address_of");
const irep_idt irept::id_and = irep_idt("and");
const irep_idt irept::id_or = irep_idt("or");
const irep_idt irept::id_array = irep_idt("array");
const irep_idt irept::id_bool = irep_idt("bool");
const irep_idt irept::id_code = irep_idt("code");
const irep_idt irept::id_constant = irep_idt("constant");
const irep_idt irept::id_dereference = irep_idt("dereference");
const irep_idt irept::id_empty = irep_idt("empty");
const irep_idt irept::id_fixedbv = irep_idt("fixedbv");
const irep_idt irept::id_floatbv = irep_idt("floatbv");
const irep_idt irept::id_incomplete_array = irep_idt("incomplete_array");
const irep_idt irept::id_index = irep_idt("index");
const irep_idt irept::id_member = irep_idt("member");
const irep_idt irept::id_not = irep_idt("not");
const irep_idt irept::id_notequal = irep_idt("notequal");
const irep_idt irept::id_pointer = irep_idt("pointer");
const irep_idt irept::id_signedbv = irep_idt("signedbv");
const irep_idt irept::id_struct = irep_idt("struct");
const irep_idt irept::id_symbol = irep_idt("symbol");
const irep_idt irept::id_typecast = irep_idt("typecast");
const irep_idt irept::id_union = irep_idt("union");
const irep_idt irept::id_unsignedbv = irep_idt("unsignedbv");
const irep_idt irept::id_vector = irep_idt("vector");
