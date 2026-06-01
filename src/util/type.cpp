#include <util/type.h>

void typet::move_to_subtypes(typet &type)
{
  subtypest &sub = subtypes();
  sub.push_back(static_cast<const typet &>(get_nil_irep()));
  sub.back().swap(type);
}

bool is_number(const typet &type)
{
  const std::string &id = type.id_string();
  return id == "complex" || id == "unsignedbv" || id == "signedbv" ||
         id == "floatbv" || id == "fixedbv";
}

bool is_array_like(const typet &type)
{
  return type.is_vector() || type.is_array() || type.is_incomplete_array();
}

irep_idt typet::t_signedbv = irep_idt("signedbv");
irep_idt typet::t_unsignedbv = irep_idt("unsignedbv");
irep_idt typet::t_complex = irep_idt("complex");
irep_idt typet::t_floatbv = irep_idt("floatbv");
irep_idt typet::t_fixedbv = irep_idt("fixedbv");
irep_idt typet::t_bool = irep_idt("bool");
irep_idt typet::t_empty = irep_idt("empty");
irep_idt typet::t_symbol = irep_idt("symbol");
irep_idt typet::t_struct = irep_idt("struct");
irep_idt typet::t_union = irep_idt("union");
irep_idt typet::t_class = irep_idt("class");
irep_idt typet::t_code = irep_idt("code");
irep_idt typet::t_array = irep_idt("array");
irep_idt typet::t_pointer = irep_idt("pointer");
irep_idt typet::t_reference = irep_idt("#reference");
irep_idt typet::t_bv = irep_idt("bv");
irep_idt typet::t_vector = irep_idt("vector");

irep_idt typet::t_intcap = irep_idt("intcap");
irep_idt typet::t_uintcap = irep_idt("uintcap");
irep_idt typet::t_ptrmem = irep_idt("ptrmem");

irep_idt typet::a_identifier = irep_idt("identifier");
irep_idt typet::a_name = irep_idt("name");
irep_idt typet::a_components = irep_idt("components");
irep_idt typet::a_methods = irep_idt("methods");
irep_idt typet::a_arguments = irep_idt("arguments");
irep_idt typet::a_return_type = irep_idt("return_type");
irep_idt typet::a_size = irep_idt("size");
irep_idt typet::a_width = irep_idt("width");
irep_idt typet::a_integer_bits = irep_idt("integer_bits");
irep_idt typet::a_f = irep_idt("f");

irep_idt typet::f_subtype = irep_idt("subtype");
irep_idt typet::f_subtypes = irep_idt("subtypes");
irep_idt typet::f_location = irep_idt("#location");
irep_idt typet::f_can_carry_provenance = irep_idt("can_carry_provenance");
