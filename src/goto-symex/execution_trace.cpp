#include <goto-symex/execution_trace.h>
#include <clang-c-frontend/expr2ccode.h>
#include <goto-programs/loopst.h>

std::string c_instructiont::convert_to_c(namespacet &ns)
{
  std::ostringstream out;

  if(loop_number > 0)
    out << "// loop " << loop_number << "\n";

  if(is_target())
    out << "__ESBMC_goto_label_" << target_number << ":;\n";
  
  switch(type)
  {
  case ASSERT:
    return convert_assert_to_c(ns);
  case NO_INSTRUCTION_TYPE:
    out << "// NO_INSTRUCTION_TYPE";
    break;
  case GOTO:
    if(!is_true(guard))
      out << "if(" << expr2ccode(migrate_expr_back(guard), ns) << ") ";

    out << "goto ";

    for(instructiont::targetst::const_iterator gt_it = targets.begin();
        gt_it != targets.end();
        gt_it++)
    {
      if(gt_it != targets.begin())
        out << ", ";
      out << "__ESBMC_goto_label_" << (*gt_it)->target_number;
    }

    out << ";";
    break;
  case FUNCTION_CALL:
    out << "// FUNCTION_CALL " << location << expr2ccode(migrate_expr_back(code), ns);
    break;
  case RETURN:
    out << "// RETURN";
    break;
  case DECL:
    out << expr2ccode(migrate_expr_back(code), ns);
    break;
  case DEAD:
    out << "// DEAD: " << expr2ccode(migrate_expr_back(code), ns);
    break;
  case OTHER:
    return convert_other_to_c(ns);
  case ASSIGN:
    out << expr2ccode(migrate_expr_back(code), ns);
    break;
  case ASSUME:
    out << "// ASSUME";
    break;
  case LOCATION:
    out << "// LOCATION";
    break;
  case THROW:
    out << "// THROW";
    break;
  case CATCH:
    out << "// CATCH";
    break;
  case ATOMIC_BEGIN:
    out << "// ATOMIC_BEGIN";
    break;
  case ATOMIC_END:
    out << "// ATOMIC_END";
    break;
  case THROW_DECL:
    out << "// THROW_DECL";
    break;
  case THROW_DECL_END:
    out << "// THROW_DECL_END";
    break;
  case SKIP:
    out << "// SKIP";
    break;
  case END_FUNCTION:
    out << "// END_FUNCTION " << location;
    break;

  default:
    throw "unknown statement";
  }
  return out.str();
}

std::string c_instructiont::convert_assert_to_c(namespacet &ns)
{
  std::ostringstream out;
  out << "assert((" << expr2ccode(migrate_expr_back(guard), ns) << ") && \"[what: " << msg <<
    "] [location: " << location << "]\");";
  return out.str();
}

std::string c_instructiont::convert_other_to_c(namespacet &ns)
{
  std::ostringstream out;
  if(is_code_printf2t(code))
  {
    out << expr2ccode(migrate_expr_back(code), ns);
    return out.str();
  }
  else
  {
    out << "// OTHER\n";
    //out << "/* " << code << " */";
  }
  return out.str();
}









