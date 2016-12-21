/*******************************************************************\

Module: Internal Representation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>
#include <assert.h>

#include "irep.h"
#include "i2string.h"
#include "string_hash.h"

irept nil_rep_storage;

#ifdef SHARING
const irept::dt empty_d;
#endif

void
irept::dump(void) const
{

  std::cout << pretty(0) << std::endl;
  return;
}

const irept &get_nil_irep()
{
  if(nil_rep_storage.id().empty()) // initialized?
    nil_rep_storage.id("nil");
  return nil_rep_storage;
}

#ifdef SHARING
irept::irept(const irep_idt &_id):data(new dt)
{
  id(_id);

  #ifdef IREP_DEBUG
  std::cout << "CREATED " << data << " " << _id << std::endl;
  #endif
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
  #ifdef IREP_DEBUG
  std::cout << "DETATCH1: " << data << std::endl;
  #endif

  if(data==NULL)
  {
    data=new dt;

    #ifdef IREP_DEBUG
    std::cout << "ALLOCATED " << data << std::endl;
    #endif
  }
  else if(data->ref_count>1)
  {
    dt *old_data(data);
    data=new dt(*old_data);

    #ifdef IREP_DEBUG
    std::cout << "ALLOCATED " << data << std::endl;
    #endif

    data->ref_count=1;
    remove_ref(old_data);
  }

  assert(data->ref_count==1);


  #ifdef IREP_DEBUG
  std::cout << "DETATCH2: " << data << std::endl;
  #endif
}
#endif

#ifdef SHARING
const irept::dt &irept::read() const
{
  #ifdef IREP_DEBUG
  std::cout << "READ: " << data << std::endl;
  #endif

  if(data==NULL)
    return empty_d;

  return *data;
}
#endif

#include <iostream>

#ifdef SHARING
void irept::remove_ref(dt *old_data)
{
  if(old_data==NULL) return;

  assert(old_data->ref_count!=0);

  #ifdef IREP_DEBUG
  std::cout << "R: " << old_data << " " << old_data->ref_count << std::endl;
  #endif

  old_data->ref_count--;
  if(old_data->ref_count==0)
  {
    #ifdef IREP_DEBUG
    std::cout << "D: " << pretty() << std::endl;
    std::cout << "DELETING " << old_data->data
              << " " << old_data << std::endl;
    old_data->clear();
    std::cout << "DEALLOCATING " << old_data << "\n";
    #endif

    delete old_data;

    #ifdef IREP_DEBUG
    std::cout << "DONE\n";
    #endif
  }
}
#endif

void irept::clear()
{
  #ifdef SHARING
  remove_ref(data);
  data=NULL;
  #else
  data.clear();
  #endif
}

void irept::move_to_named_sub(const irep_namet &name, irept &irep)
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

const irep_idt &irept::get(const irep_namet &name) const
{
  const named_subt &s=
    is_comment(name)?get_comments():get_named_sub();

  named_subt::const_iterator it=s.find(name);

  if(it==s.end())
  {
    const static irep_idt empty;
    return empty;
  }

  return it->second.id();
}

bool irept::get_bool(const irep_namet &name) const
{
  return atoi(get(name).c_str());
}

void irept::set(const irep_namet &name, const long value)
{
  add(name).id(i2string((int)value));
}

void irept::remove(const irep_namet &name)
{
  named_subt &s=
    is_comment(name)?get_comments():get_named_sub();

  named_subt::iterator it=s.find(name);

  if(it!=s.end()) s.erase(it);
}

void irept::set(const irep_namet &name, const irept &irep)
{
  add(name)=irep;
}

const irept &irept::find(const irep_namet &name) const
{
  const named_subt &s=
    is_comment(name)?get_comments():get_named_sub();

  named_subt::const_iterator it=s.find(name);

  if(it==s.end())
    return get_nil_irep();

  return it->second;
}

irept &irept::add(const irep_namet &name)
{
  named_subt &s=
    is_comment(name)?get_comments():get_named_sub();

  return s[name];
}

bool operator==(const irept &i1, const irept &i2)
{
  #ifdef SHARING
  if(i1.data==i2.data) return true;
  #endif

  if(i1.id()!=i2.id()) return false;

  if(i1.get_sub()!=i2.get_sub()) return false; // recursive call

  if(i1.get_named_sub()!=i2.get_named_sub()) return false; // recursive call

  // comments are NOT checked

  return true;
}

bool full_eq(const irept &i1, const irept &i2)
{
  #ifdef SHARING
  if(i1.data==i2.data) return true;
  #endif

  if(i1.id()!=i2.id()) return false;

  const irept::subt &i1_sub=i1.get_sub();
  const irept::subt &i2_sub=i2.get_sub();
  const irept::named_subt &i1_named_sub=i1.get_named_sub();
  const irept::named_subt &i2_named_sub=i2.get_named_sub();
  const irept::named_subt &i1_comments=i1.get_comments();
  const irept::named_subt &i2_comments=i2.get_comments();

  if(i1_sub.size()      !=i2_sub.size()) return false;
  if(i1_named_sub.size()!=i2_named_sub.size()) return false;
  if(i1_comments.size() !=i2_comments.size()) return false;

  for(unsigned i=0; i<i1_sub.size(); i++)
    if(!full_eq(i1_sub[i], i2_sub[i]))
      return false;

  {
    irept::named_subt::const_iterator i1_it=i1_named_sub.begin();
    irept::named_subt::const_iterator i2_it=i2_named_sub.begin();

    for(; i1_it!=i1_named_sub.end(); i1_it++, i2_it++)
      if(i1_it->first!=i2_it->first ||
         !full_eq(i1_it->second, i2_it->second))
        return false;
  }

  {
    irept::named_subt::const_iterator i1_it=i1_comments.begin();
    irept::named_subt::const_iterator i2_it=i2_comments.begin();

    for(; i1_it!=i1_comments.end(); i1_it++, i2_it++)
      if(i1_it->first!=i2_it->first ||
         !full_eq(i1_it->second, i2_it->second))
        return false;
  }

  return true;
}

std::string irept::to_string() const
{
  return pretty(0);
}

std::ostream& operator<< (std::ostream& out, const irept &irep)
{
  out << irep.to_string();
  return out;
}

bool ordering(const irept &X, const irept &Y)
{
  return X.compare(Y)<0;

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

  r=id().compare(i.id());
  if(r!=0) return r;

  if(get_sub().size()<i.get_sub().size()) return -1;
  if(get_sub().size()>i.get_sub().size()) return 1;

  {
    irept::subt::const_iterator it1, it2;

    for(it1=get_sub().begin(),
        it2=i.get_sub().begin();
        it1!=get_sub().end() && it2!=i.get_sub().end();
        it1++,
        it2++)
    {
      r=it1->compare(*it2);
      if(r!=0) return r;
    }

    assert(it1==get_sub().end() && it2==i.get_sub().end());
  }

  if(get_named_sub().size()<i.get_named_sub().size()) return -1;
  if(get_named_sub().size()>i.get_named_sub().size()) return 1;

  {
    irept::named_subt::const_iterator it1, it2;

    for(it1=get_named_sub().begin(),
        it2=i.get_named_sub().begin();
        it1!=get_named_sub().end() && it2!=i.get_named_sub().end();
        it1++,
        it2++)
    {
      r=it1->first.compare(it2->first);
      if(r!=0) return r;

      r=it1->second.compare(it2->second);
      if(r!=0) return r;
    }

    assert(it1==get_named_sub().end() &&
           it2==i.get_named_sub().end());
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
  size_t result=hash_string(id());

  forall_irep(it, get_sub()) result=result^it->hash();

  forall_named_irep(it, get_named_sub())
  {
    result=result^hash_string(it->first);
    result=result^it->second.hash();
  }

  return result;
}

size_t irept::full_hash() const
{
  size_t result=hash_string(id());

  forall_irep(it, get_sub()) result=result^it->full_hash();

  forall_named_irep(it, get_named_sub())
  {
    result=result^hash_string(it->first);
    result=result^it->second.full_hash();
  }

  forall_named_irep(it, get_comments())
  {
    result=result^hash_string(it->first);
    result=result^it->second.full_hash();
  }

  return result;
}

static void indent_str(std::string &s, unsigned indent)
{
  for(unsigned i=0; i<indent; i++)
    s+=' ';
}

std::string irept::pretty(unsigned indent) const
{
  std::string result;

  if(id()!="")
  {
    result+=id_string();
    indent+=2;
  }

  forall_named_irep(it, get_named_sub())
  {
    result+="\n";
    indent_str(result, indent);

    result+="* ";
    #ifdef USE_DSTRING
    result+=it->first.as_string();
    #else
    result+=it->first;
    #endif
    result+=": ";

    result+=it->second.pretty(indent+2);
  }

  forall_named_irep(it, get_comments())
  {
    result+="\n";
    indent_str(result, indent);

    result+="* ";
    #ifdef USE_DSTRING
    result+=it->first.as_string();
    #else
    result+=it->first;
    #endif
    result+=": ";

    result+=it->second.pretty(indent+2);
  }

  unsigned count=0;

  forall_irep(it, get_sub())
  {
    result+="\n";
    indent_str(result, indent);

    result+=i2string(count++);
    result+=": ";

    result+=it->pretty(indent+2);
  }

  return result;
}

const irep_idt irept::a_width = dstring("width");
const irep_idt irept::a_name = dstring("name");
const irep_idt irept::a_statement = dstring("statement");
const irep_idt irept::a_identifier = dstring("identifier");
const irep_idt irept::a_comp_name = dstring("component_name");
const irep_idt irept::a_tag = dstring("tag");
const irep_idt irept::a_from = dstring("from");
const irep_idt irept::a_file = dstring("file");
const irep_idt irept::a_line = dstring("line");
const irep_idt irept::a_function = dstring("function");
const irep_idt irept::a_column = dstring("column");
const irep_idt irept::a_access = dstring("access");
const irep_idt irept::a_destination = dstring("destination");
const irep_idt irept::a_base_name = dstring("base_name");
const irep_idt irept::a_comment = dstring("comment");
const irep_idt irept::a_event = dstring("event");
const irep_idt irept::a_literal = dstring("literal");
const irep_idt irept::a_loopid = dstring("loop-id");
const irep_idt irept::a_mode = dstring("mode");
const irep_idt irept::a_module = dstring("module");
const irep_idt irept::a_pretty_name = dstring("pretty_name");
const irep_idt irept::a_property = dstring("property");
const irep_idt irept::a_size = dstring("size");
const irep_idt irept::a_integer_bits = dstring("integer_bits");
const irep_idt irept::a_to = dstring("to");
const irep_idt irept::a_failed_symbol = dstring("#failed_symbol");
const irep_idt irept::a_dynamic = dstring("#dynamic");
const irep_idt irept::a_cmt_base_name = dstring("#base_name");
const irep_idt irept::a_id_class = dstring("#id_class");
const irep_idt irept::a_cmt_identifier = dstring("#identifier");
const irep_idt irept::a_cformat = dstring("#cformat");
const irep_idt irept::a_cmt_width = dstring("#width");
const irep_idt irept::a_axiom = dstring("axiom");
const irep_idt irept::a_cmt_constant = dstring("#constant");
const irep_idt irept::a_default = dstring("default");
const irep_idt irept::a_ellipsis = dstring("ellipsis");
const irep_idt irept::a_explicit = dstring("explicit");
const irep_idt irept::a_file_local = dstring("file_local");
const irep_idt irept::a_hex_or_oct = dstring("#hex_or_oct");
const irep_idt irept::a_hide = dstring("#hide");
const irep_idt irept::a_implicit = dstring("#implicit");
const irep_idt irept::a_incomplete = dstring("#incomplete");
const irep_idt irept::a_initialization = dstring("initialization");
const irep_idt irept::a_inlined = dstring("#inlined");
const irep_idt irept::a_invalid_object = dstring("#invalid_object");
const irep_idt irept::a_is_parameter = dstring("is_parameter");
const irep_idt irept::a_is_expression = dstring("#is_expression");
const irep_idt irept::a_is_extern = dstring("is_extern");
const irep_idt irept::a_is_macro = dstring("is_macro");
const irep_idt irept::a_is_type = dstring("is_type");
const irep_idt irept::a_cmt_lvalue = dstring("#lvalue");
const irep_idt irept::a_lvalue = dstring("lvalue");
const irep_idt irept::a_reference = dstring("#reference");
const irep_idt irept::a_restricted = dstring("#restricted");
const irep_idt irept::a_static_lifetime = dstring("static_lifetime");
const irep_idt irept::a_theorem = dstring("theorem");
const irep_idt irept::a_cmt_unsigned = dstring("#unsigned");
const irep_idt irept::a_user_provided = dstring("user-provided");
const irep_idt irept::a_cmt_volatile = dstring("#volatile");
const irep_idt irept::a_zero_initializer = dstring("#zero_initializer");
const irep_idt irept::a_flavor = dstring("flavor");
const irep_idt irept::a_cmt_active = dstring("#active");
const irep_idt irept::a_code = dstring("code");
const irep_idt irept::a_component = dstring("component");
const irep_idt irept::a_c_sizeof_type = dstring("#c_sizeof_type");
const irep_idt irept::a_end_location = dstring("#end_location");
const irep_idt irept::a_guard = dstring("guard");
const irep_idt irept::a_label = dstring("label");
const irep_idt irept::a_lhs = dstring("lhs");
const irep_idt irept::a_location = dstring("location");
const irep_idt irept::a_object_type = dstring("object_type");
const irep_idt irept::a_cmt_size = dstring("#size");
const irep_idt irept::a_cmt_type = dstring("#type");
const irep_idt irept::a_type_id = dstring("typeid");

const irep_idt irept::s_type = dstring("type");
const irep_idt irept::s_arguments = dstring("arguments");
const irep_idt irept::s_components = dstring("components");
const irep_idt irept::s_return_type = dstring("return_type");
const irep_idt irept::s_body = dstring("body");
const irep_idt irept::s_member = dstring("member");
const irep_idt irept::s_labels = dstring("labels");
const irep_idt irept::s_bv = dstring("bv");
const irep_idt irept::s_targets = dstring("targets");
const irep_idt irept::s_variables = dstring("variables");
const irep_idt irept::s_initializer = dstring("initializer");
const irep_idt irept::s_declaration_type = dstring("declaration_type");
const irep_idt irept::s_decl_value = dstring("decl_value");
const irep_idt irept::s_symvalue = dstring("symvalue");
const irep_idt irept::s_cmt_location = dstring("#location");
const irep_idt irept::s_decl_ident = dstring("decl_ident");
const irep_idt irept::s_elements = dstring("elements");
const irep_idt irept::s_offsetof_type = dstring("offsetof_type");

const irep_idt irept::id_address_of = dstring("address_of");
const irep_idt irept::id_and = dstring("and");
const irep_idt irept::id_or = dstring("or");
const irep_idt irept::id_array = dstring("array");
const irep_idt irept::id_bool = dstring("bool");
const irep_idt irept::id_code = dstring("code");
const irep_idt irept::id_constant = dstring("constant");
const irep_idt irept::id_dereference = dstring("dereference");
const irep_idt irept::id_empty = dstring("empty");
const irep_idt irept::id_fixedbv = dstring("fixedbv");
const irep_idt irept::id_floatbv = dstring("floatbv");
const irep_idt irept::id_incomplete_array = dstring("incomplete_array");
const irep_idt irept::id_index = dstring("index");
const irep_idt irept::id_member = dstring("member");
const irep_idt irept::id_not = dstring("not");
const irep_idt irept::id_notequal = dstring("notequal");
const irep_idt irept::id_pointer = dstring("pointer");
const irep_idt irept::id_signedbv = dstring("signedbv");
const irep_idt irept::id_struct = dstring("struct");
const irep_idt irept::id_symbol = dstring("symbol");
const irep_idt irept::id_typecast = dstring("typecast");
const irep_idt irept::id_union = dstring("union");
const irep_idt irept::id_unsignedbv = dstring("unsignedbv");
