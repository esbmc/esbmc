/*******************************************************************\

Module: goto_programt -> irep conversion

Author: CM Wintersteiger

Date: May 2007

\*******************************************************************/

#include <i2string.h>

#include "goto_program_irep.h"

void convert(const goto_programt::instructiont &instruction, irept &irep)
{
  irep.code(migrate_expr_back(instruction.code));

  if (instruction.function!="")
    irep.function(instruction.function);

  if (instruction.location.is_not_nil())
    irep.location(instruction.location);

  irep.type_id((long) instruction.type);

  irep.guard(migrate_expr_back(instruction.guard));

  if(! instruction.targets.empty())
  {
    irept tgts;
    for(goto_programt::targetst::const_iterator it=
          instruction.targets.begin();
        it!=instruction.targets.end();
        it++)
    {
      irept t(i2string((*it)->location_number));
      tgts.move_to_sub(t);
    }

    irep.targets(tgts);
  }

  if(! instruction.labels.empty())
  {
    irept lbls;
    irept::subt &subs = lbls.get_sub();
    subs.reserve(instruction.labels.size());
    for(goto_programt::instructiont::labelst::const_iterator it=
          instruction.labels.begin();
        it!=instruction.labels.end();
        it++)
    {
      subs.push_back(irept(*it));
    }

    irep.labels(lbls);
  }

  if (! instruction.local_variables.empty())
  {
    irept vars;
    irept::subt &subs = vars.get_sub();
    subs.reserve(instruction.local_variables.size());
    for(goto_programt::local_variablest::const_iterator it=
            instruction.local_variables.begin();
        it!=instruction.local_variables.end();
        it++)
    {
      subs.push_back(irept(*it));
    }

    irep.variables(vars);
  }
}

void convert(const irept &irep, goto_programt::instructiont &instruction)
{
  migrate_expr(static_cast<const exprt&>(irep.code()), instruction.code);
  migrate_expr(static_cast<const exprt&>(irep.guard()), instruction.guard);
  instruction.function = irep.function_irep().id();
  instruction.location = static_cast<const locationt&>(irep.location());
  instruction.type = static_cast<goto_program_instruction_typet>(
                  atoi(irep.type_id().c_str()));

  // don't touch the targets, the goto_programt conversion does that

  const irept &lbls=irep.labels_irep();
  const irept::subt &lsubs=lbls.get_sub();
  for (irept::subt::const_iterator it=lsubs.begin();
       it!=lsubs.end();
       it++)
  {
    instruction.labels.push_back(it->id());
  }

  const irept &vars=irep.variables();
  const irept::subt &vsubs=vars.get_sub();
  for (irept::subt::const_iterator it=vsubs.begin();
       it!=vsubs.end();
       it++)
  {
    instruction.local_variables.insert(it->id());
  }
}

void convert( const goto_programt &program, irept &irep )
{
  irep.id("goto-program");
  irep.get_sub().reserve(program.instructions.size());
  for (goto_programt::instructionst::const_iterator it=
          program.instructions.begin();
       it!=program.instructions.end();
       it++)
  {
    irep.get_sub().push_back(irept());
    convert(*it, irep.get_sub().back());
  }
}

void convert( const irept &irep, goto_programt &program )
{
  assert(irep.id()=="goto-program");

  program.instructions.clear();

  std::list< std::list<unsigned> > number_targets_list;

  // convert instructions back
  const irept::subt &subs = irep.get_sub();
  for (irept::subt::const_iterator it=subs.begin();
       it!=subs.end();
       it++)
  {
    program.instructions.push_back(goto_programt::instructiont());
    convert(*it, program.instructions.back());

    number_targets_list.push_back(std::list<unsigned>());
    const irept &targets=it->targets();
    const irept::subt &tsubs=targets.get_sub();
    for (irept::subt::const_iterator tit=tsubs.begin();
         tit!=tsubs.end();
         tit++)
    {
      number_targets_list.back().push_back(
          atoi(tit->id_string().c_str()));
    }
  }

  program.compute_location_numbers();

  // resolve targets
  std::list< std::list<unsigned> >::iterator nit=
        number_targets_list.begin();
  for(goto_programt::instructionst::iterator lit=
        program.instructions.begin();
      lit!=program.instructions.end() && nit!=number_targets_list.end();
      lit++, nit++)
  {
    for (std::list<unsigned>::iterator tit=nit->begin();
         tit!=nit->end();
         tit++)
    {
      goto_programt::targett fit=program.instructions.begin();
      for(;fit!=program.instructions.end();fit++)
      {
        if (fit->location_number==*tit)
        {
          lit->targets.push_back(fit);
          break;
        }
      }

      if (fit==program.instructions.end())
      {
        std::cout << "Warning: could not resolve target link " <<
        "during irep->goto_program translation." << std::endl;
        throw 0;
      }
    }
  }

  program.update();
}
