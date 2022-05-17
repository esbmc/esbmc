#include <goto-programs/goto_domain.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>

void print_free_vars(std::vector<expr2tc> &vars, const messaget &msg)
{
  std::stringstream ss;
  ss << "Free variables found: ";
  for(auto v : vars)
  {
    symbol2t s = to_symbol2t(v);

    std::vector<std::string> sv;
    boost::split(sv, s.thename.as_string(), boost::is_any_of("@"));

    ss << sv.back() << " ";
  }
  msg.status(ss.str());
}

//gets free variable declared in the main function
//these are the candidates for domain splitting
std::vector<expr2tc>
get_free_vars_main(goto_functionst &goto_functions)
{
  std::vector<expr2tc> free_vars;

  irep_idt main_idt(
    "c:@F@main"); // domain split on a variable in the main function

  goto_functionst::function_mapt &funmap = goto_functions.function_map;
  goto_functiont &main = funmap.at(main_idt);

  if(main.body_available)
  {
    goto_programt &body = main.body;
    //Get the instructions stored in the goto program
    goto_programt::instructionst &ins = body.instructions;

    auto i = ins.begin();
    for(; i != ins.end(); i++)
    {
      if(i->is_assign())
      {
        //convert to assignment
        const code_assign2t &ass = to_code_assign2t(i->code);

        if(ass.source->expr_id != expr2t::symbol_id)
          continue;

        //const symbol2t &lhs = to_symbol2t(ass.target);
        const symbol2t &rhs = to_symbol2t(ass.source);
        /* msg.status(fmt::format( */
        /*   "{} == {}", lhs.thename.as_string(), rhs.thename.as_string())); */

        //check if rhs is non-det variables
        //note that if rhs is not symbol the thename variable is empty
        std::string nd("_nondet_");
        if(boost::algorithm::contains(rhs.thename.as_string(), nd))
        {
          //add lhs (which we have determined is a free variable) to the vector
          free_vars.push_back(ass.target);
        }
      }
    }
  }

  return free_vars;
}

void goto_assume_cond(goto_programt &goto_p, expr2tc &cond)
{
  goto_programt::targett e = goto_p.add_instruction(ASSUME);
  e->guard = static_cast<expr2tc>(cond);
  e->inductive_step_instruction = false;
  e->inductive_assertion = false;
  e->location.comment("domain-partition");
}

//inserts a assumption into the goto program to split the domain
void goto_domain_split_numeric(
  goto_programt &goto_program,
  const symbol2tc &var,
  const uint32_t val,
  const bool lt
  )
{
  auto &ins = goto_program.instructions;
  goto_programt::targett i = ins.begin();

  //seek first assignment for the variable to split the domain on
  for(; i != ins.end(); i++)
  {
    if(i->is_assign())
    {
      const code_assign2t &ass = to_code_assign2t(i->code);

      if(ass.source->expr_id != expr2t::symbol_id)
        continue;

      const symbol2t &lhs = to_symbol2t(ass.target);
      /* const symbol2t &rhs = to_symbol2t(ass.source); */
      /* msg.status(fmt::format( */
      /*   "{} == {}", lhs.thename.as_string(), rhs.thename.as_string())); */

      if(lhs.thename.as_string() == var->thename.as_string())
      {
        i++;
        break;
      }
    }
  }

  /* expr2tc var = free_vars.front(); */

  /* msg.status(fmt::format("{} {}", var->expr_id, var.use_count())); */

  //create internal representation of condition things
  const expr2tc c = constant_int2tc(get_int32_type(), val);
  const lessthan2tc condlt(var, c);
  const greaterthanequal2tc condgt(var, c);

  expr2tc cond;

  // Should we split top half or bottom half
  if(lt)
  {
    cond = static_cast<expr2tc>(condlt);
  }
  else
  {
    cond = static_cast<expr2tc>(condgt);
  }

  // Create a temporary goto_program which will be inserted into

  goto_programt::targett e = goto_program.insert(i);
  e->type = ASSUME;
  e->guard = static_cast<expr2tc>(cond);
  e->inductive_step_instruction = false;
  e->inductive_assertion = false;
  e->location.comment("domain-partition");

  /* i++; */

  //insert assumption after the first assignment of the variable to its non-det value.
  /* goto_program.insert_swap(i, tmp_e); */
}

