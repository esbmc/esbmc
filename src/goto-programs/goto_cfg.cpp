#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <goto-programs/goto_cfg.h>

goto_cfg::goto_cfg(goto_functionst &goto_functions)
{
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    goto_programt &body = f_it->second.body;

    // First pass - identify all the leaders
    std::set<goto_programt::instructionst::iterator> leaders;
    leaders.insert(
      body.instructions.begin()); // First instruction is always a leader

    Forall_goto_program_instructions (i_it, body)
    {
      if (i_it->is_target())
      {
        leaders.insert(i_it);
      }

      if (i_it->is_goto() || i_it->is_backwards_goto())
      {
        for (const auto &target : i_it->targets)
          leaders.insert(target);

        auto next = i_it;
        next++;
        leaders.insert(next);
      }

      if (i_it->is_return())
      {
        auto next = i_it;
        next++;
        leaders.insert(next);
      }

      if (i_it->is_throw() || i_it->is_catch())
      {
        log_error("[CFG], Throw and catch instructions are not supported yet");
        abort();
      }

      // TODO: there are some special C functions that should be handled: exit, longjmp, etc.
    }

    // Second pass - identify all the basic blocks
    auto start = body.instructions.begin();
    const auto &end = body.instructions.end();
    std::vector<std::shared_ptr<basic_block>> bbs;

    while (start != end)
    {
      std::shared_ptr<basic_block> bb = std::make_shared<basic_block>();
      bb->begin = start;
      start++;
      bb->end = start;

      while (start != end && leaders.find(start) == leaders.end())
      {
        start++;
        bb->end = start;
      }

      if (bbs.size() > 0)
      {
        bbs.back()->successors.insert(bb);
        bb->predecessors.insert(bbs.back());
      }
      bbs.push_back(bb);
    }

    // Third pass - identify all the successors/predecessors
    for (auto &bb : bbs)
    {
      assert(bb->begin != bb->end);
      auto last = bb->end;
      last--;

      for (const auto &bb2 : bbs)
      {
        if (bb2->begin == bb->end)
        {
          bb->successors.insert(bb2);
          bb2->predecessors.insert(bb);
        }

        if (last->is_goto() || last->is_backwards_goto())
        {
          bb->terminator = basic_block::terminator_type::IF_GOTO;
          for (const auto &target : last->targets)
          {
            if (target == bb2->begin)
            {
              bb->successors.insert(bb2);
              bb2->predecessors.insert(bb);
            }
          }
        }
      }
    }

    basic_blocks[f_it->first.as_string()] = bbs;
  }

  log_progress("Finished CFG construction");
}

#include <fstream>

void goto_cfg::dump_graph() const
{
  log_status("Dumping CFG");
  for (const auto &[function, bbs] : basic_blocks)
  {
    std::ofstream file("cfg_" + function + ".dot");
    file << "digraph G {\n";
    for (size_t t = 0; t < bbs.size(); t++)
    {
      file << "BB" << t << " [shape=record, label=\"{" << t << ":\\l|";
      for (auto i = bbs[t]->begin; i != bbs[t]->end; i++)
      {
        std::ostringstream oss;
        i->output_instruction(*migrate_namespace_lookup, "", oss);
        // TODO: escape special characters
        file << oss.str() << "\\l";
      }

      switch (bbs[t]->terminator)
      {
      case basic_block::terminator_type::IF_GOTO:
      {
        file << "|{<s0>T|<s1>F}}\"];\n";
        auto suc = bbs[t]->successors.begin();
        file << "BB" << t << ":s0"
             << " -> "
             << "BB"
             << std::distance(
                  bbs.begin(), std::find(bbs.begin(), bbs.end(), *suc))
             << ";\n";
        suc++;
        file << "BB" << t << ":s1"
             << " -> "
             << "BB"
             << std::distance(
                  bbs.begin(), std::find(bbs.begin(), bbs.end(), *suc))
             << ";\n";
      }
      break;

      default:
        file << "}\"];\n";
        for (const auto &suc : bbs[t]->successors)
          file << "BB" << t << " -> "
               << "BB"
               << std::distance(
                    bbs.begin(), std::find(bbs.begin(), bbs.end(), suc))
               << ";\n";
        break;
      }
    }
    file << "}\n";
  }
}

void rec_replace_symbols(
  expr2tc &e,
  const std::unordered_map<std::string, expr2tc> &constant_expressions)
{
  if (!e)
    return;

  if (is_symbol2t(e))
  {
    auto it = constant_expressions.find(to_symbol2t(e).thename.as_string());
    if (it != constant_expressions.end())
    {
      e = it->second;
      return;
    }
  }

  e->Foreach_operand(
                     [&constant_expressions](expr2tc &e2) { rec_replace_symbols(e2, constant_expressions); });

}


void GOTO_CFG_OPTIMIZATIONS::apply_bb_constant_folding(goto_functionst &goto_functions)
{
  goto_cfg cfg(goto_functions);
  for (auto &[function, bbs] : cfg.basic_blocks)
  {
    log_status("[CFG] Constant folding in function {}", function);
    
    for (auto bb : bbs)
    {
      std::unordered_map<std::string , expr2tc> constant_expressions;
      for (auto it = bb->begin; it != bb->end; it++)
      {
        if (it->is_assert() || it->is_assume() || it->is_backwards_goto() || it->is_goto())
        {
          // Optimize condition
          rec_replace_symbols(it->guard, constant_expressions);
          simplify(it->guard);
        }
        else if (it->is_decl())
        {
          // Update map
          constant_expressions.erase(to_code_decl2t(it->code).value.as_string());
        }
        else if (it->is_assign())
        {
          // Optimize RHS and update map
          code_assign2t &assign = to_code_assign2t(it->code);

          rec_replace_symbols(assign.source, constant_expressions);
          simplify(assign.source);
          if (is_dereference2t(assign.target))
          {
            // Just clear until we have a points-to
            constant_expressions.clear();
          }
          else if (is_symbol2t(assign.target))
          {
            symbol2t &sym = to_symbol2t(assign.target);
            if (is_constant(assign.source)) {
              constant_expressions[sym.thename.as_string()] = assign.source;
                }
            else {
              constant_expressions.erase(sym.thename.as_string());
}
          }
          else
          {
            log_warning("[CFG] Unsupported assignment");
            assign.target->dump();
          }
        }
        else if (it->is_function_call())
        {
          // Optimize arguments
          rec_replace_symbols(it->code, constant_expressions);
          simplify(it->code);
           
        }
        else if (it->is_return())
        {
          // Optimize result
          rec_replace_symbols(it->code, constant_expressions);
          simplify(it->code);
        }
        else if (it->is_throw() || it->is_catch())
        {
          log_error("Constant folding does not support try/cath");
          abort();
        }
      }
    }
      
  }
  
}
