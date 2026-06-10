#include <ld-frontend/ld_language.h>
#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <ld-frontend/semantics/type_checker.h>
#include <ld-frontend/ir/ld_ir_builder.h>
#include <ld-frontend/ir_gen/ld_converter.h>
#include <ld-frontend/property/yaml_property_parser.h>
#include <ld-frontend/property/property_encoder.h>
#include <util/c_expr2string.h>
#include <util/config.h>
#include <util/message.h>
#include <iostream>

languaget *new_ld_language()
{
  return new ld_languaget;
}

bool ld_languaget::parse(const std::string &path)
{
  log_debug("ld", "Parsing: {}", path);

  // When driven by the esbmc CLI (not ld-verify), pick up the property file
  // from the --ld-props option unless one was already set explicitly.
  if (props_path_.empty())
    props_path_ = config.options.get_option("ld-props");

  try
  {
    PlcopenXmlParser parser;
    ast_ = parser.parse(path);
  }
  catch (const UnsupportedConstructError &e)
  {
    log_error("{}", e.what());
    return true;
  }
  catch (const LdParseError &e)
  {
    log_error("{}", e.what());
    return true;
  }

  try
  {
    TypeChecker checker;
    checker.check(ast_);
  }
  catch (const TypeCheckError &e)
  {
    log_error("{}", e.what());
    return true;
  }

  if (!props_path_.empty())
  {
    try
    {
      YamlPropertyParser prop_parser;
      props_ = prop_parser.parse(props_path_);
    }
    catch (const LdPropertyParseError &e)
    {
      log_error("{}", e.what());
      return true;
    }
  }

  return false;
}

bool ld_languaget::typecheck(contextt &context, const std::string & /*module*/)
{
  try
  {
    LdIRBuilder builder;
    LdIR ir = builder.build(ast_);

    ld_converter converter(context, ir);
    converter.convert();

    // Append property assertions into the scan-loop body of ld::scan_loop.
    if (!props_.empty())
    {
      property_encoder encoder(context, ast_.source_file);
      code_blockt prop_code = encoder.encode(props_);

      symbolt *scan_sym = context.find_symbol("ld::scan_loop");
      if (scan_sym && !prop_code.operands().empty())
      {
        // scan_loop body is a code_blockt whose only operand is the while loop.
        exprt scan_val = scan_sym->get_value();
        if (
          !scan_val.operands().empty() &&
          to_code(scan_val.operands().front()).get_statement() == "while")
        {
          codet &loop = static_cast<codet &>(scan_val.operands().front());
          code_whilet &whl = static_cast<code_whilet &>(loop);
          for (const auto &op : prop_code.operands())
            whl.body().copy_to_operands(op);
        }
        scan_sym->set_value(scan_val);
      }
    }

    // Emit static init after all symbols (including property counters) exist.
    converter.prepend_static_init();
  }
  catch (const std::runtime_error &e)
  {
    log_error("{}", e.what());
    return true;
  }

  return false;
}

bool ld_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = c_expr2string(expr, ns, flags);
  return false;
}

bool ld_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = c_type2string(type, ns, flags);
  return false;
}

unsigned ld_languaget::default_flags(presentationt target) const
{
  unsigned f = 0;
  switch (target)
  {
  case presentationt::HUMAN:
    f |= c_expr2stringt::SHORT_ZERO_COMPOUNDS;
    break;
  case presentationt::WITNESS:
    f |= c_expr2stringt::UNIQUE_FLOAT_REPR;
    break;
  }
  return f;
}

void ld_languaget::show_parse(std::ostream &out)
{
  out << "LD program: " << ast_.source_file << "\n";
  out << "Variables: " << ast_.variables.size() << "\n";
  for (const auto &net : ast_.networks)
  {
    out << "Network '" << net.name << "': " << net.rungs.size() << " rungs\n";
    for (const auto &rung : net.rungs)
      out << "  Rung " << rung.id << ": " << rung.elements.size()
          << " elements\n";
  }
}
