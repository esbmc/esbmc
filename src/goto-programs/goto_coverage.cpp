#include <goto-programs/goto_coverage.h>
void goto_coverage(goto_functionst goto_functions)
{
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
        goto_programt& goto_program = f_it->second.body;
        Forall_goto_program_instructions(it, goto_program)
        {
            // make_assertion_false
            if((*it).is_assert())
            {
                log_status("make_assertion_false");
                (*it).guard = gen_false_expr();
                it++;
            }
            
            // add_false_assertion
            if((*it).is_return() || (*it).is_end_function())
            {
                log_status("add_false_assertion");
                goto_programt::targett t = goto_program.insert(it);
                t->type = ASSERT;
                t->guard = gen_false_expr();
                t->location = (*it).location;
                it++;
            }
        }
    }
}