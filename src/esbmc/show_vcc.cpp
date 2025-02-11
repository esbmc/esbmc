#include <esbmc/bmc.h>
#include <fmt/format.h>
#include <iostream>

#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <util/migrate.h>

void bmct::show_vcc(std::ostream &out, const symex_target_equationt &eq)
{
  out << "\nVERIFICATION CONDITIONS:\n\n";

  languagest languages(ns, language_idt::C);

  for (symex_target_equationt::SSA_stepst::const_iterator it =
         eq.SSA_steps.begin();
       it != eq.SSA_steps.end();
       it++)
  {
    if (!it->is_assert() || it->ignore)
      continue;

    if (it->source.pc->location.is_not_nil())
      out << it->source.pc->location << "\n";

    if (it->comment != "")
      out << it->comment << "\n";

    symex_target_equationt::SSA_stepst::const_iterator p_it =
      eq.SSA_steps.begin();

    for (unsigned count = 1; p_it != it; p_it++)
      if (p_it->is_assume() || p_it->is_assignment())
        if (!p_it->ignore)
        {
          std::string string_value;
          languages.from_expr(migrate_expr_back(p_it->cond), string_value);
          out << "{-" << count << "} " << string_value << "\n";
          count++;
        }

    out << "|--------------------------"
        << "\n";

    std::string string_value;
    languages.from_expr(migrate_expr_back(it->cond), string_value);
    out << "{" << 1 << "} " << string_value << "\n";

    out << "\n";
  }
}

void bmct::show_vcc(const symex_target_equationt &eq)
{
  const std::string &filename = options.get_option("output");

  if (filename.empty() || filename == "-")
  {
    show_vcc(std::cout, eq);
  }
  else
  {
    std::ofstream out(filename.c_str());
    if (!out)
      log_error("failed to open {}", filename);
    else
      show_vcc(out, eq);
  }
}
