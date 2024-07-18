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

irep_idt typet::t_signedbv = dstring("signedbv");
irep_idt typet::t_unsignedbv = dstring("unsignedbv");
irep_idt typet::t_complex = dstring("complex");
irep_idt typet::t_floatbv = dstring("floatbv");
irep_idt typet::t_fixedbv = dstring("fixedbv");
irep_idt typet::t_bool = dstring("bool");
irep_idt typet::t_empty = dstring("empty");
irep_idt typet::t_symbol = dstring("symbol");
irep_idt typet::t_struct = dstring("struct");
irep_idt typet::t_union = dstring("union");
irep_idt typet::t_class = dstring("class");
irep_idt typet::t_code = dstring("code");
irep_idt typet::t_array = dstring("array");
irep_idt typet::t_pointer = dstring("pointer");
irep_idt typet::t_reference = dstring("#reference");
irep_idt typet::t_bv = dstring("bv");
irep_idt typet::t_vector = dstring("vector");

irep_idt typet::t_intcap = dstring("intcap");
irep_idt typet::t_uintcap = dstring("uintcap");

irep_idt typet::a_identifier = dstring("identifier");
irep_idt typet::a_name = dstring("name");
irep_idt typet::a_components = dstring("components");
irep_idt typet::a_methods = dstring("methods");
irep_idt typet::a_arguments = dstring("arguments");
irep_idt typet::a_return_type = dstring("return_type");
irep_idt typet::a_size = dstring("size");
irep_idt typet::a_width = dstring("width");
irep_idt typet::a_integer_bits = dstring("integer_bits");
irep_idt typet::a_f = dstring("f");

irep_idt typet::f_subtype = dstring("subtype");
irep_idt typet::f_subtypes = dstring("subtypes");
irep_idt typet::f_location = dstring("#location");
irep_idt typet::f_can_carry_provenance = dstring("can_carry_provenance");
