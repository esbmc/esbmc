#include <memory>

#include <util/language.h>
#include <langapi/mode.h>
#include <langapi/language_util.h>

#include "show_symbol_table.h"

void show_symbol_table_xml_ui()
{
}

void show_symbol_table_plain(const namespacet &ns, std::ostream &out)
{
  out << "\n"
      << "Symbols:"
      << "\n";
  out << "Number of symbols: " << ns.get_context().size() << "\n";
  out << "\n";

  ns.get_context().foreach_operand_in_order([&out, &ns](const symbolt &s) {
    std::unique_ptr<languaget> p = language_from_symbol(s);
    std::string type_str, value_str;

    if (s.type.is_not_nil())
      p->from_type(s.type, type_str, ns);

    if (s.value.is_not_nil())
      p->from_expr(s.value, value_str, ns);

    out << "Symbol......: " << s.id << "\n";
    out << "Module......: " << s.module << "\n";
    out << "Base name...: " << s.name << "\n";
    out << "Mode........: " << s.mode << "\n";
    out << "Type........: " << type_str << "\n";
    out << "Value.......: " << value_str << "\n";
    out << "Flags.......:";

    if (s.lvalue)
      out << " lvalue";
    if (s.static_lifetime)
      out << " static_lifetime";
    if (s.file_local)
      out << " file_local";
    if (s.is_type)
      out << " type";
    if (s.is_extern)
      out << " extern";
    if (s.is_macro)
      out << " macro";

    out << "\n";
    out << "Location....: " << s.location << "\n";

    out << "\n";
  });
}
