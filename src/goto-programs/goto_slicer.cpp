#include <goto-programs/goto_slicer.h>

namespace {
// Recursively try to extract the nondet symbol of an expression
void get_symbols(const expr2tc &expr, std::set<std::string> &values)
{
  // TODO: This function should return a set!
  switch(expr->expr_id)
  {
  case expr2t::symbol_id:
    values.insert(to_symbol2t(expr).get_symbol_name());
    return;

  case expr2t::with_id:
    return get_symbols(to_with2t(expr).update_value, values);

  case expr2t::byte_extract_id:
    return get_symbols(to_byte_extract2t(expr).source_value, values);

  case expr2t::typecast_id:
    return get_symbols(to_typecast2t(expr).from, values);

  case expr2t::bitcast_id:
    return get_symbols(to_bitcast2t(expr).from, values);

  case expr2t::constant_struct_id:
    for(const auto &v : to_constant_struct2t(expr).datatype_members)
    {
      get_symbols(v, values);
    }
    return;

  case expr2t::if_id:
  {
    get_symbols(to_if2t(expr).true_value, values);
    get_symbols(to_if2t(expr).false_value, values);
    return;
  }
  default:

    return;
  }
}
}


void slicer_domaint::output(std::ostream &out) const
{
  if(is_bottom())
  {
    out << "BOTTOM\n";
    return;
  }

  out << "\t[ ";
  for(const auto &d : dependencies)
  {

    out << "\n\t\t" << d.first << ": {";
    for(const auto &s : d.second)
    {
      out << " " << s << " ";
    }
    out << d.first << " }\n\t";
  }

  out << "]";
}

void slicer_domaint::assign(const expr2tc &e) {
  auto assignment = to_code_assign2t(e);

  // We don't care about constants
  if(!is_symbol2t(assignment.target))
    return;

  std::set<std::string> vars;
  get_symbols(assignment.source, vars);
  // Some magic to get the vars
  dependencies[to_symbol2t(assignment.target).get_symbol_name()] = vars;
}

void slicer_domaint::declaration(const expr2tc &e) {

  auto A = to_code_decl2t(e);

  std::set<std::string> vars;
  // Some magic to get the vars
  dependencies[A.value.as_string()] = vars;
}

void slicer_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case DECL:
    declaration(instruction.code);
    break;

  case ASSIGN:
    assign(instruction.code);
    break;

  case GOTO:
  {

  }
  break;

  case ASSUME:

    break;

  case FUNCTION_CALL:
  {
  }
  break;


  default:;
  }
}

bool slicer_domaint::join(const slicer_domaint &b)
{
  if(b.is_bottom())
    return false;

  bool changed = false;
  for(auto &d : b.dependencies)
  {
    if(!dependencies.count(d.first))
    {
      dependencies[d.first] = d.second;
      changed = true;
    }

    else {
      for(auto &x : d.second)
        if(dependencies[d.first].count(x) == 0)
        {
          dependencies[d.first].insert(x);
          changed = true;
        }
    }
  }
  return changed;
}

bool goto_slicer::runOnProgram(goto_functionst &F)
{
  sl(F, ns);
  return true;
}

#include <boost/range/adaptor/reversed.hpp>
bool goto_slicer::runOnFunction(std::pair<const dstring, goto_functiont> &F) {
  if(F.first != "c:@F@main") return false;
  if(F.second.body_available)
  {
    log_status("//// Function: {}\n", F.first);
    for(goto_programt::instructionst::iterator it =
          --(F.second.body).instructions.end();
        it != (F.second.body).instructions.begin();
        it--)
    {
      switch(it->type)
      {
      case DECL:
        break;

      case ASSIGN:
        break;

      case GOTO:
      break;

      case ASSUME:
        break;

      case FUNCTION_CALL:
      break;

      case ASSERT:
        it->make_skip();
        break;


      default:;
      }
    }

  }

  return true;
}
