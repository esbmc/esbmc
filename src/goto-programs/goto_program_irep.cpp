/*******************************************************************
 Module: goto_programt -> irep conversion

 Author: CM Wintersteiger

 Date: May 2007

\*******************************************************************/

#include <goto-programs/goto_program_irep.h>
#include <util/i2string.h>

void convert(const goto_programt::instructiont &instruction, irept &irep)
{
  irep.code(migrate_expr_back(instruction.code));

  if(instruction.function != "")
    irep.function(instruction.function);

  if(instruction.location.is_not_nil())
    irep.location(instruction.location);

  irep.type_id((long)instruction.type);

  irep.guard(migrate_expr_back(instruction.guard));

  if(!instruction.targets.empty())
  {
    irept tgts;
    for(auto const &it : instruction.targets)
    {
      irept t(i2string(it->location_number));
      tgts.move_to_sub(t);
    }

    irep.targets(tgts);
  }

  if(!instruction.labels.empty())
  {
    irept lbls;
    irept::subt &subs = lbls.get_sub();
    subs.reserve(instruction.labels.size());
    for(auto const &it : instruction.labels)
      subs.emplace_back(it);

    irep.labels(lbls);
  }
}

void convert(const irept &irep, goto_programt::instructiont &instruction)
{
  migrate_expr(static_cast<const exprt &>(irep.code()), instruction.code);
  migrate_expr(static_cast<const exprt &>(irep.guard()), instruction.guard);
  instruction.function = irep.function_irep().id();
  instruction.location = static_cast<const locationt &>(irep.location());
  instruction.type =
    static_cast<goto_program_instruction_typet>(atoi(irep.type_id().c_str()));

  // don't touch the targets, the goto_programt conversion does that

  const irept &lbls = irep.labels_irep();
  const irept::subt &lsubs = lbls.get_sub();
  for(auto const &it : lsubs)
    instruction.labels.push_back(it.id());
}

void convert(const goto_programt &program, irept &irep)
{
  irep.id("goto-program");
  irep.get_sub().reserve(program.instructions.size());
  for(auto const &it : program.instructions)
  {
    irep.get_sub().emplace_back();
    convert(it, irep.get_sub().back());
  }

  irep.hide(program.hide);
}

void convert(const irept &irep, goto_programt &program)
{
  assert(irep.id() == "goto-program");

  program.instructions.clear();

  std::list<std::list<unsigned>> number_targets_list;

  // convert instructions back
  const irept::subt &subs = irep.get_sub();
  for(auto const &it : subs)
  {
    program.instructions.emplace_back();
    convert(it, program.instructions.back());

    number_targets_list.emplace_back();
    const irept &targets = it.targets();
    const irept::subt &tsubs = targets.get_sub();
    for(auto const &tit : tsubs)
      number_targets_list.back().push_back(atoi(tit.id_string().c_str()));
  }

  program.compute_location_numbers();

  // resolve targets
  std::list<std::list<unsigned>>::iterator nit = number_targets_list.begin();
  for(goto_programt::instructionst::iterator lit = program.instructions.begin();
      lit != program.instructions.end() && nit != number_targets_list.end();
      lit++, nit++)
  {
    for(unsigned int &tit : *nit)
    {
      goto_programt::targett fit = program.instructions.begin();
      for(; fit != program.instructions.end(); fit++)
      {
        if(fit->location_number == tit)
        {
          lit->targets.push_back(fit);
          break;
        }
      }

      if(fit == program.instructions.end())
      {
        program.msg.error(
          "Warning: could not resolve target link "
          "during irep->goto_program translation.");
        abort();
      }
    }
  }

  program.update();

  program.hide = irep.hide();
}
