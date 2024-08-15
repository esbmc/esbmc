#include "c_types.h"
#include "irep2/irep2_expr.h"
#include "irep2/irep2_utils.h"
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
    int counter = 0;
    for (auto &bb : bbs)
    {
      assert(bb->begin != bb->end);
      auto last = bb->end;
      last--;
      bb->uuid = counter++;

      for (const auto &bb2 : bbs)
      {
        const bool is_goto = last->is_goto() || last->is_backwards_goto();
        const bool is_trivial_goto = is_goto && is_true(last->guard);
        if (bb2->begin == bb->end)
        {
          if (is_trivial_goto)
          {
            bb->successors.erase(bb2);
            bb2->predecessors.erase(bb);
          }
          else
          {
            bb->successors.insert(bb2);
            bb2->predecessors.insert(bb);
          }
        }

        if (is_goto)
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
        if (bbs[t]->successors.size() == 1)
        {
          file << "}\"];\n";
          for (const auto &suc : bbs[t]->successors)
            file << "BB" << t << " -> "
                 << "BB"
                 << std::distance(
                      bbs.begin(), std::find(bbs.begin(), bbs.end(), suc))
                 << ";\n";
          break;
        }
        else
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

template <class F>
void goto_cfg::basic_block::foreach_inst(F f)
{
  for (goto_programt::instructionst::iterator start = begin; start != end;
       start++)
    f(*start);
}

template <class F>
void goto_cfg::basic_block::foreach_bb(F f)
{
  goto_cfg::foreach_bb(this, f);
}

template <class F>
void goto_cfg::foreach_bb(
  const std::shared_ptr<goto_cfg::basic_block> &start,
  F foo)
{
  std::unordered_set<std::shared_ptr<goto_cfg::basic_block>> visited;
  std::vector<std::shared_ptr<goto_cfg::basic_block>> to_visit({start});

  while (!to_visit.empty())
  {
    const auto &item = to_visit.back();
    visited.insert(item);
    to_visit.pop_back();
    foo(item);
    for (const auto &next : item->successors)
      if (!visited.count(next))
        to_visit.push_back(next);
  }
}

void Dominator::compute_dominators()
{
  // TODO: This algorithm is quadratic, there is a more efficient
  // version by Lengauer-Tarjan.

  // Computes dominator of a node
  std::unordered_map<CFGNode, std::unordered_set<CFGNode>> dt;

  // 1. Sets all nodes dominators to TOP

  std::unordered_set<CFGNode> top;
  goto_cfg::foreach_bb(start, [&top](const CFGNode &n) { top.insert(n); });

  goto_cfg::foreach_bb(start, [&dt, &top](const CFGNode &n) { dt[n] = top; });

  /*
    Dom(n) = {n}, if n = start
             {n} `union` (intersec_pred Dom(p))
   */
  dt[start] = std::unordered_set<CFGNode>({start});

  std::vector<std::shared_ptr<goto_cfg::basic_block>> worklist{start};
  for (const auto &suc : start->successors)
    worklist.push_back(suc);

  while (!worklist.empty())
  {
    const auto &dominator_node = worklist.back();
    worklist.pop_back();

    if (dominator_node == start)
      continue;

    // Get the intersection of all the predecessors
    std::unordered_set<CFGNode> intersection = top;
    for (const auto &pred : dominator_node->predecessors)
    {
      assert(dt.count(pred));
      for (const auto &item : top)
        if (!dt[pred].count(item))
          intersection.erase(item);
    }

    // Union of node with its predecessor intersection
    std::unordered_set<CFGNode> result({dominator_node});
    for (const auto &item : intersection)
      result.insert(item);

    // Fix-point?
    if (dt[dominator_node] != result)
    {
      dt[dominator_node] = result;
      for (const auto &suc : dominator_node->successors)
        worklist.push_back(suc);
    }
  }

  _dominators = dt;
}

void Dominator::dump_dominators() const
{
  for (const auto &[node, edges] : _dominators)
  {
    log_status("CFGNode");
    node->begin->dump();

    log_status("Dominated");
    for (const auto &edge : edges)
      edge->begin->dump();
  }
}

CFGNode Dominator::idom(const CFGNode &n) const
{
  std::vector<CFGNode> sdoms;
  const auto &n_dominators = dom(n);
  std::copy_if(
    n_dominators.begin(),
    n_dominators.end(),
    std::back_inserter(sdoms),
    [&n, this](const auto &item) { return sdom(item, n); });

  for (const CFGNode &n0 : sdoms)
  {
    bool valid_result = true;
    for (const CFGNode &n1 : sdoms)
    {
      if (sdom(n0, n1))
      {
        valid_result = false;
        break;
      }
    }
    if (valid_result)
      return n0;
  }
  log_error("[cfg] Unable to compute `idom`");
  abort();
}

void Dominator::dump_idoms() const
{
  DomTree dt(*this);

  log_status("Root: {}", dt.root->uuid);
  for (const auto &[key, value] : dt.edges)
  {
    log_status("CFGNode: {}", key->uuid);
    log_status("Edges");
    for (const auto &n : value)
      log_status("\t{}", n->uuid);
  }
}

std::unordered_set<CFGNode> Dominator::dom_frontier(const CFGNode &n) const
{
  assert(dj);
  std::unordered_set<CFGNode> result;
  const auto levels = dj->tree.get_levels();
  for (const CFGNode &y : dj->tree.get_subtree(n))
  {
    for (const CFGNode &z : dj->_jedges[y])
      if (levels.at(z) <= levels.at(n))
        result.insert(z);
  }
  return result;
}

std::unordered_set<CFGNode>
Dominator::dom_frontier(const std::unordered_set<CFGNode> &nodes) const
{
  assert(nodes.size() > 0);

  std::unordered_set<CFGNode> result;

  for (const CFGNode &n : nodes)
    for (const CFGNode &df : dom_frontier(n))
      result.insert(df);

  return result;
}

std::unordered_set<CFGNode>
Dominator::iterated_dom_frontier(const std::unordered_set<CFGNode> &nodes) const
{
  assert(dj);
  assert(nodes.size() > 0);
  // TODO: There is a linear solution by SREEDHAR using PiggyBanks
  std::unordered_set<CFGNode> result = dom_frontier(nodes);
  std::unordered_set<CFGNode> result2 = dom_frontier(result);

  while (1)
  {
    result = result2;
    result2 = dom_frontier(result);
    assert(result2.size() >= result.size());

    if (result.size() != result2.size())
      continue;

    for (const auto &n : result)
      if (!result2.count(n))
        continue;

    break;
  }

  return result;
}

Dominator::DJGraph::DJGraph(const CFGNode &cfg, const Dominator &dom)
  : tree(dom)
{
  // A DJ-Graph is a graph composed by D-Edges and J-Edges
  _graph = tree.edges;
  _dedges = tree.edges;
  auto func = [this, &dom](const CFGNode &x) {
    auto [val, ins] = _graph.insert({x, std::unordered_set<CFGNode>()});
    for (const CFGNode &y : x->successors)
      if (!dom.sdom(x, y))
        val->second.insert(y);
    _jedges[x] = val->second;
  };
  goto_cfg::foreach_bb(cfg, func);
}

void Dominator::DJGraph::DJGraph::dump() const
{
  log_status("Dumping DJ-Graph");
  for (const auto &[k, edges] : _graph)
  {
    log_status("CFGNode {}", k->uuid);
    for (const auto &e : edges)
      log_status("\t->{}", e->uuid);
  }
}

std::unordered_map<CFGNode, size_t> Dominator::DomTree::get_levels() const
{
  std::unordered_map<CFGNode, size_t> levels;
  levels[root] = 0;

  std::set<CFGNode> worklist({root});
  while (!worklist.empty())
  {
    CFGNode current = *worklist.begin();
    worklist.erase(current);
    if (!edges.count(current))
      continue;

    size_t level = levels[current] + 1;
    for (const CFGNode &nodes : edges.at(current))
    {
      levels[nodes] = level;
      worklist.insert(nodes);
    }
  }
  return levels;
}

std::unordered_set<CFGNode>
Dominator::DomTree::get_subtree(const CFGNode &n) const
{
  std::unordered_set<CFGNode> nodes({n});
  std::set<CFGNode> worklist({n});

  while (!worklist.empty())
  {
    const CFGNode current = *worklist.begin();
    worklist.erase(current);

    if (!edges.count(current))
      continue;

    for (const auto &inner : edges.at(current))
    {
      worklist.insert(inner);
      nodes.insert(inner);
    }
  }

  return nodes;
}

Dominator::DomTree::DomTree(const Dominator &dom) : root(dom.start)
{
  goto_cfg::foreach_bb(dom.start, [this, &dom](const CFGNode &n) {
    if (n == root)
      return;

    CFGNode root = dom.idom(n);
    auto [val, ins] = edges.insert({root, std::unordered_set<CFGNode>()});
    val->second.insert(n);
  });
}

void ssa_promotion::promote()
{
  // TODO: The promote_node algorithm works by:
  // 1. Identify and compute the liveness of each var (read-only)
  // 2. Insert the phi nodes (affects all instructions)
  // 3. Rename the phi nodes (affects per var instructions)
  // We could process 1 and 3 in parallel
  for (auto &[k, v] : cfg.basic_blocks)
  {
    if (_skip.count(k))
      continue;

    assert(v.size());
    promote_node(goto_functions.function_map[k].body, v[0]);
  }
  goto_functions.update();
}

void replace_symbols_in_expr(
  expr2tc &use,
  symbolt *symbol,
  unsigned last_id_in_bb,
  const irep_idt var_name)
{
  if (!use)
    return;

  if (is_phi2t(use))
    return;

  if (is_symbol2t(use) && to_symbol2t(use).thename == var_name)
  {
    use = symbol2tc(use->type, symbol->id);
    return;
  }

  use->Foreach_operand([symbol, last_id_in_bb, var_name](expr2tc &e) {
    replace_symbols_in_expr(e, symbol, last_id_in_bb, var_name);
  });
}

std::unordered_map<std::string, ssa_promotion::symbol_information>
ssa_promotion::extract_symbol_information(const CFGNode &start) const
{
  std::unordered_map<std::string, symbol_information> symbol_map;
  std::unordered_set<std::string> unpromotable_symbols;

  const auto insert_definition = [&symbol_map](
                                   const irep_idt &var,
                                   goto_programt::instructionst::iterator &I,
                                   const CFGNode &bb) {
    // Are we dealing with a global?
    if (!symbol_map.count(var.as_string()))
      return;
    ;
    symbol_information &sym = symbol_map[var.as_string()];

    sym.def_blocks.insert(bb);
    sym.def_instructions.insert(
      std::make_shared<goto_programt::instructiont>(*I));
  };

  const auto insert_use = [&symbol_map, &unpromotable_symbols](
                            const expr2tc &e,
                            goto_programt::instructionst::iterator &I,
                            const CFGNode &bb) {
    std::unordered_set<expr2tc, irep2_hash> symbols;
    get_addr_symbols(e, symbols);
    for (const expr2tc &sym_expr : symbols)
    {
      assert(is_symbol2t(sym_expr));
      const symbol2t &symbol = to_symbol2t(sym_expr);
      unpromotable_symbols.insert(symbol.thename.as_string());
    }

    symbols.clear();
    get_symbols(e, symbols);
    for (const expr2tc &sym_expr : symbols)
    {
      assert(is_symbol2t(sym_expr));
      const symbol2t &symbol = to_symbol2t(sym_expr);
      assert(symbol_map.count(symbol.thename.as_string()));
      symbol_information &sym = symbol_map[symbol.thename.as_string()];

      // TODO: When using live analysis, only consider a use-block if the value is used before redefined
      sym.use_blocks.insert(bb);
      sym.use_instructions.insert(
        std::make_shared<goto_programt::instructiont>(*I));
    }
  };

  goto_cfg::foreach_bb(
    start,
    [this, &insert_definition, &insert_use, &symbol_map](const CFGNode &bb) {
      for (goto_programt::instructionst::iterator it = bb->begin; it != bb->end;
           it++)
      {
        if (it->is_decl())
        {
          symbol_information info;
          const code_decl2t &decl = to_code_decl2t(it->code);
          info.type = decl.type;
          symbol_map.insert({decl.value.as_string(), info});
          const symbolt *const symbol = context.find_symbol(decl.value);
          assert(symbol != nullptr);
          info.mode = symbol->mode.as_string();

          // TODO: I am assuming that all declarations are followed by
          //       an init assignment. So decls definitions are useless
        }

        else if (it->is_assign())
        {
          const expr2tc &target = to_code_assign2t(it->code).target;
          if (is_symbol2t(target))
            insert_definition(to_symbol2t(target).thename, it, bb);

          insert_use(to_code_assign2t(it->code).source, it, bb);
        }
        else if (it->is_function_call())
          for (const auto &op : to_code_function_call2t(it->code).operands)
            insert_use(op, it, bb);
        else
        {
          insert_use(it->code, it, bb);
          insert_use(it->guard, it, bb);
        }
      }
    });

  for (auto symbol : unpromotable_symbols)
    symbol_map.erase(symbol);

  // TODO: Technically, not all address uses needs to be avoided
  // TODO: Some memsets/memcpy could be replaced with values
  return symbol_map;
}

void ssa_promotion::promote_node(goto_programt &P, const CFGNode &n)
{
  Dominator dom(n);

  /**
   * Promoting a node to SSA means adding all the phi-nodes for each promotoble
   * variable. It consists in:
   *
   * 1. Identifying information for all variables in the function: name, type, definitions, uses.
   * 2. Compute the phi-nodes.
   * 3. Rename all uses and definitions. This consists in numbering the variables accordingly and fixing the
   *    phi-node connections
   */

  auto symbol_map = extract_symbol_information(n);

  for (auto &[var, info] : symbol_map)
  {
    // Some symbols might come from globals or functions
    if (info.def_blocks.empty() || !info.type)
    {
      log_warning("Could not promote {}", var);
      continue;
    }

    const auto phinodes = dom.dom_frontier(info.def_blocks);

    // Lets add all symbols at the beginning
    std::vector<symbolt *> symbols;
    std::unordered_map<std::string, unsigned> symbol_to_index;
    for (size_t i = 0; i < info.def_instructions.size() + phinodes.size(); i++)
    {
      symbolt symbol;
      symbol.type = migrate_type_back(info.type);
      symbol.id = fmt::format("{}.{}", var, i);
      symbol.name = symbol.id;
      symbol.mode = "C"; // TODO: get mode
      symbol.location = n->begin->location;

      symbolt *symbol_in_context = context.move_symbol_to_context(symbol);
      assert(symbol_in_context != nullptr);
      symbols.push_back(symbol_in_context);
      symbol_to_index[symbol_in_context->id.as_string()] = i;

      goto_programt::instructiont decl;
      decl.make_decl();
      decl.code = code_decl2tc(info.type, symbol_in_context->id);
      decl.location = n->begin->location;
      P.insert_swap(n->begin, decl);
    }

    unsigned counter = 0;
    std::unordered_map<size_t, CFGNode>
      ref_to_nodes; // the cfg locations are no longer reliable
    // Insert phi-nodes
    for (const CFGNode &phinode : phinodes)
    {
      if (phinode->predecessors.size() != 2)
      {
        log_error(
          "Constructed a phi-node with a value different than 2. Please check "
          "the goto-cfg construction and the phinode placement algorithm");
        abort();
      }
      goto_programt::instructiont phi_instr;
      phi_instr.make_assignment();
      phi_instr.location = phinode->begin->location;
      phi_instr.function = phinode->begin->function;
      auto phi_symbol = symbol2tc(info.type, symbols[counter++]->id);

      const expr2tc lhs = symbol2tc(info.type, symbols[0]->id);
      const expr2tc rhs = symbol2tc(info.type, symbols[0]->id);
      auto pred_it = phinode->predecessors.begin();

      locationt lhs_location;
      locationt rhs_location;

      {
        auto before_end = (*pred_it)->end;
        before_end--;
        lhs_location = before_end->location;
        ref_to_nodes[lhs_location.hash()] = *pred_it;
      }
      pred_it++;

      {
        auto before_end = (*pred_it)->end;
        before_end--;
        rhs_location = before_end->location;
        ref_to_nodes[rhs_location.hash()] = *pred_it;
      }
      const expr2tc phi_expr =
        phi2tc(info.type, lhs, rhs, lhs_location, rhs_location);
      phi_instr.code = code_assign2tc(phi_symbol, phi_expr);

      P.insert_swap(phinode->begin, phi_instr);
      info.def_blocks.insert(phinode);
      info.use_blocks.insert(phinode);
    }
    std::unordered_map<CFGNode, unsigned> last_id_in_bb;

    // For each definition rename the symbol.
    for (auto defBlock : info.def_blocks)
    {
      for (auto instruction = defBlock->begin; instruction != defBlock->end;
           instruction++)
      {
        if (!instruction->is_assign())
          continue;

        const auto code_expr = to_code_assign2t(instruction->code);
        if (!(is_symbol2t(code_expr.target) &&
              to_symbol2t(code_expr.target).thename == var))
          continue;

        last_id_in_bb[defBlock] = counter;
        auto new_symbol = symbol2tc(info.type, symbols[counter++]->id);
        instruction->code = code_assign2tc(new_symbol, code_expr.source);
      }
    }

    auto get_last_defined = [&last_id_in_bb](const CFGNode n) -> unsigned {
      // Back search
      std::set<CFGNode> visited;
      std::vector<CFGNode> worklist{n};
      while (!worklist.empty())
      {
        const CFGNode item = worklist.back();
        worklist.pop_back();

        if (!visited.insert(item).second)
          continue;

        if (last_id_in_bb.count(item))
          return last_id_in_bb[item];

        for (auto &pred : item->predecessors)
          worklist.push_back(pred);
      }
      log_error("Could not find when variable was last defined");
      abort();
    };

    auto find_location = [&get_last_defined, &ref_to_nodes, &symbols](
                           const CFGNode, unsigned id) -> symbolt * {
      return symbols[get_last_defined(ref_to_nodes[id])];
    };

    // For each use rename the symbol
    for (auto useBlock : info.use_blocks)
    {
      unsigned current_last_id = 0;
      for (auto instruction = useBlock->begin; instruction != useBlock->end;
           instruction++)
      {
        if (
          instruction->is_decl() &&
          to_code_decl2t(instruction->code).value == var)
        {
          instruction->make_skip();
          instruction->code = code_skip2tc(int_type2());
        }

        else if (instruction->is_assign())
        {
          auto assign = to_code_assign2t(instruction->code);
          assert(current_last_id <= symbols.size());

          if (!is_phi2t(assign.source))
            replace_symbols_in_expr(
              assign.source, symbols[current_last_id], 0, var);
          else
          {
            // TODO set location correctly
            auto lhs_symbol = find_location(
              useBlock, to_phi2t(assign.source).lhs_location.hash());
            auto rhs_symbol = find_location(
              useBlock, to_phi2t(assign.source).rhs_location.hash());

            if (!lhs_symbol || !rhs_symbol)
              continue;

            if (!has_prefix(
                  to_symbol2t(assign.target).thename.as_string(), var))
              continue;

            assert(has_prefix(lhs_symbol->id, var));
            assert(has_prefix(rhs_symbol->id, var));

            to_phi2t(assign.source).lhs = symbol2tc(info.type, lhs_symbol->id);
            to_phi2t(assign.source).rhs = symbol2tc(info.type, rhs_symbol->id);
          }
          if (
            is_symbol2t(assign.target) &&
            symbol_to_index.count(
              to_symbol2t(assign.target).thename.as_string()))
            current_last_id = symbol_to_index.at(
              to_symbol2t(assign.target).thename.as_string());

          instruction->code = code_assign2tc(assign.target, assign.source);
        }
        else
        {
          replace_symbols_in_expr(
            instruction->code, symbols[current_last_id], 0, var);
          replace_symbols_in_expr(
            instruction->guard, symbols[current_last_id], 0, var);
        }
      }
    }
    // No need to kill the new symbols as they are always memsafe
  }
}
