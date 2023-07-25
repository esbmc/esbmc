#include <goto-programs/goto_functions.h>

void goto_location(goto_functionst &goto_functions, std::string src_fl)
{
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_function_call() && it->location.is_not_nil())
        {
          //! only consider the situation that the caller is in the source file
          std::string caller_fl = it->location.file().as_string();
          if(caller_fl != src_fl)
            continue;

          code_function_call2t &function_call =
            to_code_function_call2t(it->code);

          // Don't do function pointers
          if(is_dereference2t(function_call.function))
            continue;

          // find code in function map
          irep_idt &identifier = to_symbol2t(function_call.function).thename;
          goto_functionst::function_mapt::const_iterator iter =
            goto_functions.function_map.find(identifier);

          // skip if the body is not found or empty
          if(iter == goto_functions.function_map.end())
            continue;

          const goto_functiont &goto_function = iter->second;
          if(!goto_function.body_available)
            continue;

          std::string callee_fl =
            goto_function.body.instructions.front().location.file().as_string();

          // this means the function body is defined in another file
          if(caller_fl != callee_fl)
          {
            // add two false asserts that contains location info
            // e.g.
            // assert(0); // atom_start
            // func_call();
            // assert(0); // atom_end

            goto_programt::targett t = goto_program.insert(it);
            t->type = ASSERT;
            t->guard = gen_false_expr();
            t->location = it->location;
            t->location.property("atom_start");
            t->location.comment("show caller location");
            t->location.user_provided(true);
            it = ++t;

            it++; // no out of range risk due to END_FUNCTION
            t = goto_program.insert(it);
            t->type = ASSERT;
            t->guard = gen_false_expr();
            t->location = it->location;
            t->location.property("atom_end");
            t->location.comment("show caller location");
            t->location.user_provided(true);
            it = t;
          }
        }
      }
    }
}
