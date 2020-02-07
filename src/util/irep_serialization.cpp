/*******************************************************************\

Module: binary irep conversions with hashing

Author: CM Wintersteiger

Date: May 2007

\*******************************************************************/

#include <sstream>
#include <util/irep_serialization.h>
#include "irep2_type.h"
#include "irep2_expr.h"

void irep_serializationt::write_irep(std::ostream &out, const irept &irep)
{
  write_string_ref(out, irep.id_string());

  forall_irep(it, irep.get_sub())
  {
    out.put('S');
    reference_convert(*it, out);
  }

  forall_named_irep(it, irep.get_named_sub())
  {
    out.put('N');
    write_string_ref(out, name2string(it->first));
    reference_convert(it->second, out);
  }

  forall_named_irep(it, irep.get_comments())
  {
    out.put('C');
    write_string_ref(out, name2string(it->first));
    reference_convert(it->second, out);
  }

  out.put(0); // terminator
}

void irep_serializationt::reference_convert(std::istream &in, irept &irep)
{
  unsigned id = read_long(in);

  if(
    ireps_container.ireps_on_read.find(id) !=
    ireps_container.ireps_on_read.end())
  {
    irep = ireps_container.ireps_on_read[id];
  }
  else
  {
    read_irep(in, irep);
    ireps_container.ireps_on_read[id] = irep;
  }
}

void irep_serializationt::read_irep(std::istream &in, irept &irep)
{
  irep.id(read_string_ref(in));

  while(in.peek() == 'S')
  {
    in.get();
    irep.get_sub().emplace_back();
    reference_convert(in, irep.get_sub().back());
  }

  while(in.peek() == 'N')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  while(in.peek() == 'C')
  {
    in.get();
    irept &r = irep.add(read_string_ref(in));
    reference_convert(in, r);
  }

  if(in.get() != 0)
  {
    std::cerr << "irep not terminated. " << std::endl;
    throw 0;
  }
}

void irep_serializationt::reference_convert(
  const irept &irep,
  std::ostream &out)
{
  // Do we have this irep already? Horrible complexity here.
  unsigned int i;
  for(i = 0; i < ireps_container.ireps_on_write.size(); i++)
  {
    if(full_eq(ireps_container.ireps_on_write[i], irep))
    {
      // Match, at idx i
      write_long(out, i);
      return;
    }
  }

  i = ireps_container.ireps_on_write.size();
  ireps_container.ireps_on_write.push_back(irep);
  write_long(out, i);
  write_irep(out, irep);
}

void irep_serializationt::write_long(std::ostream &out, unsigned u)
{
  out.put((u & 0xFF000000) >> 24);
  out.put((u & 0x00FF0000) >> 16);
  out.put((u & 0x0000FF00) >> 8);
  out.put(u & 0x000000FF);
}

unsigned irep_serializationt::read_long(std::istream &in)
{
  unsigned res = 0;

  for(unsigned i = 0; i < 4 && in.good(); i++)
    res = (res << 8) | in.get();

  return res;
}

void irep_serializationt::write_string(std::ostream &out, const std::string &s)
{
  for(char i : s)
  {
    if(i == 0 || i == '\\')
      out.put('\\'); // escape specials
    out << i;
  }

  out.put(0);
}

dstring irep_serializationt::read_string(std::istream &in)
{
  char c;
  unsigned i = 0;
  std::vector<char> read_buffer;
  read_buffer.resize(1, 0);

  while((c = in.get()) != 0)
  {
    if(i >= read_buffer.size())
      read_buffer.resize(read_buffer.size() * 2, 0);
    if(c == '\\') // escaped chars
      read_buffer[i] = in.get();
    else
      read_buffer[i] = c;
    i++;
  }

  if(i >= read_buffer.size())
    read_buffer.resize(read_buffer.size() * 2, 0);
  read_buffer[i] = 0;

  return dstring(&(read_buffer[0]));
}

void irep_serializationt::write_string_ref(std::ostream &out, const dstring &s)
{
  unsigned id = s.get_no();
  if(id >= ireps_container.string_map.size())
    ireps_container.string_map.resize(id + 1, false);

  if(ireps_container.string_map[id])
    write_long(out, id);
  else
  {
    ireps_container.string_map[id] = true;
    write_long(out, id);
    write_string(out, s.as_string());
  }
}

irep_idt irep_serializationt::read_string_ref(std::istream &in)
{
  unsigned id = read_long(in);

  if(id >= ireps_container.string_rev_map.size())
    ireps_container.string_rev_map.resize(
      1 + id * 2, std::pair<bool, dstring>(false, dstring()));
  if(ireps_container.string_rev_map[id].first)
  {
    return ireps_container.string_rev_map[id].second;
  }

  dstring s = read_string(in);
  ireps_container.string_rev_map[id] = std::pair<bool, dstring>(true, s);
  return ireps_container.string_rev_map[id].second;
}

/*
** SPECIALIZATIONS FOR EVERYTHING IREP2 RELATED
*/

// irep_containert

template <>
void type2tc::serialize(std::ostream &os)
{
  os << "type2tc";
}

template <>
std::shared_ptr<irep_serializable> type2tc::create(std::istream &in)
{
  return std::make_shared<type2tc>();
}

template <>
void expr2tc::serialize(std::ostream &os)
{
  os << "expr2tc";
}

template <>
std::shared_ptr<irep_serializable> expr2tc::create(std::istream &in)
{
  return std::make_shared<expr2tc>();
}

template <>
std::shared_ptr<type2tc> type2tc::unserialize(std::istream &in)
{
  static std::map<std::string, std::unique_ptr<type2tc>> subclasses;
  subclasses["type2tc"] = std::make_unique<type2tc>();

  std::string classname;
  in >> classname;
  if(subclasses.find(classname) == subclasses.end())
    throw std::bad_cast();

  std::shared_ptr<irep_serializable> ptr = subclasses[classname]->create(in);
  return std::static_pointer_cast<type2tc>(ptr);
}

template <>
std::shared_ptr<expr2tc> expr2tc::unserialize(std::istream &in)
{
  static std::map<std::string, std::unique_ptr<expr2tc>> subclasses;
  subclasses["expr2tc"] = std::make_unique<expr2tc>();

  std::string classname;
  in >> classname;
  if(subclasses.find(classname) == subclasses.end())
    throw std::bad_cast();

  std::shared_ptr<irep_serializable> ptr = subclasses[classname]->create(in);
  return std::static_pointer_cast<expr2tc>(ptr);
}

// type2t

void type2t::serialize(std::ostream &os)
{
  unsigned value = (unsigned)this->type_id;
  os << value;
}

std::shared_ptr<irep_serializable> bool_type2t::create(std::istream &in)
{
  return std::make_shared<bool_type2t>();
}

std::shared_ptr<irep_serializable> empty_type2t::create(std::istream &in)
{
  return std::make_shared<empty_type2t>();
}

std::shared_ptr<irep_serializable> symbol_type2t::create(std::istream &in)
{
  return std::make_shared<symbol_type2t>("asd");
}

std::shared_ptr<irep_serializable> struct_type2t::create(std::istream &in)
{
  std::vector<type2tc> type_vec;
  std::vector<irep_idt> memb_vec;
  std::vector<irep_idt> memb_pretty_vec;
  irep_idt name;
  return std::make_shared<struct_type2t>(
    type_vec, memb_vec, memb_pretty_vec, name, false);
}

std::shared_ptr<irep_serializable> union_type2t::create(std::istream &in)
{
  std::vector<type2tc> type_vec;
  std::vector<irep_idt> memb_vec;
  std::vector<irep_idt> memb_pretty_vec;
  irep_idt name;

  return std::make_shared<union_type2t>(
    type_vec, memb_vec, memb_pretty_vec, name);
}

std::shared_ptr<irep_serializable> unsignedbv_type2t::create(std::istream &in)
{
  return std::make_shared<unsignedbv_type2t>(1);
}

std::shared_ptr<irep_serializable> signedbv_type2t::create(std::istream &in)
{
  return std::make_shared<signedbv_type2t>(1);
}

std::shared_ptr<irep_serializable> code_type2t::create(std::istream &in)
{
  std::vector<type2tc> vec;
  type2tc type;
  std::vector<irep_idt> names;
  return std::make_shared<code_type2t>(vec, type, names, false);
}

std::shared_ptr<irep_serializable> array_type2t::create(std::istream &in)
{
  type2tc t_container;
  expr2tc e_container;
  return std::make_shared<array_type2t>(t_container, e_container, false);
}

std::shared_ptr<irep_serializable> pointer_type2t::create(std::istream &in)
{
  type2tc subtype;
  return std::make_shared<pointer_type2t>(subtype);
}

std::shared_ptr<irep_serializable> fixedbv_type2t::create(std::istream &in)
{
  return std::make_shared<fixedbv_type2t>(1, 1);
}

std::shared_ptr<irep_serializable> floatbv_type2t::create(std::istream &in)
{
  return std::make_shared<floatbv_type2t>(1, 1);
}

std::shared_ptr<irep_serializable> string_type2t::create(std::istream &in)
{
  return std::make_shared<string_type2t>(1);
}

std::shared_ptr<irep_serializable> cpp_name_type2t::create(std::istream &in)
{
  irep_idt irep;
  std::vector<type2tc> vec;
  return std::make_shared<cpp_name_type2t>(irep, vec);
}

std::shared_ptr<irep_serializable> type2t::create(std::istream &in)
{
  throw std::runtime_error("type2t is never directly created");
}

std::shared_ptr<type2t> type2t::unserialize(std::istream &in)
{
  static bool initialized = false;
  static std::map<type2t::type_ids, std::unique_ptr<type2t>> subclasses;

  if(!initialized)
  {
    const irep_idt irep_idt_obj;
    const std::vector<type2tc> vec_type2tc;
    const type2tc container_type2t;
    const expr2tc container_expr2t;
    const std::vector<irep_idt> vec_irep_idt;

    subclasses[type2t::type_ids::bool_id] = std::make_unique<bool_type2t>();
    subclasses[type2t::type_ids::cpp_name_id] =
      std::make_unique<cpp_name_type2t>(irep_idt_obj, vec_type2tc);
    subclasses[type2t::type_ids::string_id] =
      std::make_unique<string_type2t>(0);
    subclasses[type2t::type_ids::floatbv_id] =
      std::make_unique<floatbv_type2t>(0, 1);
    subclasses[type2t::type_ids::fixedbv_id] =
      std::make_unique<fixedbv_type2t>(0, 1);
    subclasses[type2t::type_ids::pointer_id] =
      std::make_unique<pointer_type2t>(container_type2t);
    subclasses[type2t::type_ids::array_id] =
      std::make_unique<array_type2t>(container_type2t, container_expr2t, true);
    subclasses[type2t::type_ids::code_id] = std::make_unique<code_type2t>(
      vec_type2tc, container_type2t, vec_irep_idt, true);
    subclasses[type2t::type_ids::signedbv_id] =
      std::make_unique<signedbv_type2t>(0);
    subclasses[type2t::type_ids::unsignedbv_id] =
      std::make_unique<unsignedbv_type2t>(0);
    subclasses[type2t::type_ids::empty_id] = std::make_unique<empty_type2t>();
    subclasses[type2t::type_ids::union_id] = std::make_unique<union_type2t>(
      vec_type2tc, vec_irep_idt, vec_irep_idt, irep_idt_obj);
    subclasses[type2t::type_ids::struct_id] = std::make_unique<struct_type2t>(
      vec_type2tc, vec_irep_idt, vec_irep_idt, irep_idt_obj);
    subclasses[type2t::type_ids::symbol_id] =
      std::make_unique<symbol_type2t>("");

    initialized = true;
  }

  unsigned classname;
  in >> classname;
  if(subclasses.find((type2t::type_ids)classname) == subclasses.end())
    throw std::bad_cast();

  std::shared_ptr<irep_serializable> ptr =
    subclasses[(type2t::type_ids)classname]->create(in);
  return std::dynamic_pointer_cast<type2t>(ptr);
}

// expr2t

std::shared_ptr<irep_serializable> expr2t::create(std::istream &in)
{
  throw std::runtime_error("expr2t is never directly created");
}

std::shared_ptr<expr2t> expr2t::unserialize(std::istream &in)
{
  static bool initialized = false;
  static std::map<expr2t::expr_ids, std::unique_ptr<expr2t>> subclasses;

  if(!initialized)
  {
    type2tc tc;
    subclasses[expr2t::expr_ids::unknown_id] = std::make_unique<unknown2t>(tc);
    initialized = true;
  }

  unsigned classname;
  in >> classname;
  if(subclasses.find((expr2t::expr_ids)classname) == subclasses.end())
    throw std::bad_cast();

  std::shared_ptr<irep_serializable> ptr =
    subclasses[(expr2t::expr_ids)classname]->create(in);

  return std::dynamic_pointer_cast<expr2t>(ptr);
}

void expr2t::serialize(std::ostream &os)
{
  auto id = (unsigned)this->expr_id;
  os << id;
}

std::shared_ptr<irep_serializable> unknown2t::create(std::istream &in)
{
  type2tc tc;
  return std::make_shared<unknown2t>(tc);
}

std::shared_ptr<irep_serializable> extract2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}
<<<<<<< Updated upstream

std::shared_ptr<irep_serializable> concat2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> popcount2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bswap2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> signbit2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> isfinite2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> isnormal2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> isinf2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable>
code_cpp_throw_decl_end2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable>
code_cpp_throw_decl2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_cpp_throw2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_cpp_catch2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_cpp_delete2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable>
code_cpp_del_array2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_asm2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> invalid_pointer2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_comma2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable>
code_function_call2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> object_descriptor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_goto2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_free2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_skip2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_return2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_expression2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_printf2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_dead2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_decl2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_init2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_assign2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> code_block2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> sideeffect2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> dynamic_size2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> deallocated_obj2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> valid_object2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> dereference2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> dynamic_object2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> null_object2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> invalid2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> overflow_neg2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> overflow_cast2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> overflow2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> isnan2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> index2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> member2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> with2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> byte_update2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> address_of2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> pointer_object2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> pointer_offset2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> same_object2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ashr2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> shl2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> modulus2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_sqrt2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_fma2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_div2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_mul2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_sub2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> ieee_add2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> div2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> mul2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> sub2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> add2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> abs2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> neg2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> lshr2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitnot2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitnxor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitnor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitnand2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitxor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitand2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> implies2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> xor2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> or2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> and2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> not2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> greaterthanequal2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> lessthanequal2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> greaterthan2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> lessthan2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> notequal2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> equality2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> if2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> bitcast2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> typecast2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> nearbyint2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> symbol2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_array_of2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_array2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_union2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_struct2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_string2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_bool2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}

std::shared_ptr<irep_serializable> constant_floatbv2t::create(std::istream &in)
{
  throw std::runtime_error("Not implemented");
}
=======
>>>>>>> Stashed changes
