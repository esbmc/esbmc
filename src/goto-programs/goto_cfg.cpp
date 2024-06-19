#include <goto-programs/goto_cfg.h>

goto_cfg::goto_cfg(goto_functionst &goto_functions)
{
  log_status("Building CFG");
  Forall_goto_functions (f_it, goto_functions)
  {
    std::shared_ptr<basic_block> function_start = std::make_shared<basic_block>();
    functions[f_it->first.as_string()] = function_start;
    if(!f_it->second.body_available)
        continue;

    goto_programt &body = f_it->second.body;
    function_start->begin = body.instructions.begin();

    // First pass - identify all the labels
    std::unordered_map<size_t, std::shared_ptr<basic_block>> labels;
    log_progress("Building labels for function: {} ", f_it->first.as_string());
    Forall_goto_program_instructions (i_it, body)
    {
      if (i_it->is_target())
      {
        log_progress("Adding label {} ", i_it->target_number);
        labels[i_it->target_number] = std::make_shared<basic_block>();
        labels[i_it->target_number]->begin = i_it;
      }
    }
    log_progress("Finished building labels");

    // Second pass - identify all the basic blocks
    std::shared_ptr<basic_block> current_block = function_start;
    Forall_goto_program_instructions (i_it, body)
    {
      if (i_it->is_target())
      {
        assert(labels.find(i_it->target_number) != labels.end());
        auto tmp = i_it;
        tmp--; // the label itself is not part of the block
        current_block->end = tmp;

        const std::shared_ptr<basic_block> &label_start = labels[i_it->target_number];
        current_block->successors.insert(label_start);
        label_start->predecessors.insert(current_block);

        current_block = label_start;
      }
      else if (i_it->is_goto())
      {
        assert(labels.find(i_it->target_number) != labels.end());
        current_block->end = i_it;
        const std::shared_ptr<basic_block> &label_start = labels[i_it->target_number];
        current_block->successors.insert(label_start);
        label_start->predecessors.insert(current_block);

        if(0 && i_it->guard)
        {
            // If the guard is false, we go to the next instruction
            std::shared_ptr<basic_block> next_block = std::make_shared<basic_block>();
            current_block->successors.insert(next_block);
            next_block->predecessors.insert(current_block);

            auto tmp = i_it;
            tmp++;
            next_block->begin = tmp;
            current_block = next_block;
        }
      }
    }
  }
  log_progress("Finished CFG construction");

}

void goto_cfg::dump_graph() const {
    log_status("Dumping CFG");

    for (const auto& [function, bb] : functions)
    {
        log_status("Function: {}", function);

        std::unordered_set<std::shared_ptr<basic_block>> visited;
        std::unordered_set<std::shared_ptr<basic_block>> to_visit = {bb};

        while(!to_visit.empty())
        {
            std::shared_ptr<basic_block> current = *to_visit.begin();
            visited.insert(current);
            to_visit.erase(to_visit.begin());

            log_status("Basic Block:");
            for(goto_programt::instructionst::iterator it = current->begin; it != current->end; it++)
            {
                it->dump();
            }

            for(const auto &successor : current->successors)
            {
                log_status("Successor: {}", std::to_string(successor->begin->location_number));
                if(visited.find(successor) == visited.end())
                    to_visit.insert(successor);
            }

            for(const auto &successor : current->predecessors)
            {
                log_status("Predecessor: {}", std::to_string(successor->begin->location_number));
                if(visited.find(successor) == visited.end())
                    to_visit.insert(successor);
            }
        }
    }

}