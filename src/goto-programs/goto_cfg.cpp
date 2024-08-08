#include "irep2/irep2_expr.h"
#include "std_code.h"
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

        // Is this an IF? 
        if (!is_true(i_it->guard))
        {          
          auto next = i_it;
          next++;
          leaders.insert(next);          
        }
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
        if (bb2->begin == bb->end)
        {
          bb->successors.insert(bb2);
          bb2->predecessors.insert(bb);
        }

        if ((last->is_goto() || last->is_backwards_goto()) && !is_true(last->guard))
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
        file << "BB" << t << ":s0" << " -> " << "BB"
             << std::distance(
                  bbs.begin(), std::find(bbs.begin(), bbs.end(), *suc))
             << ";\n";
        suc++;
        file << "BB" << t << ":s1" << " -> " << "BB"
             << std::distance(
                  bbs.begin(), std::find(bbs.begin(), bbs.end(), *suc))
             << ";\n";
      }
      break;

      default:
        file << "}\"];\n";
        for (const auto &suc : bbs[t]->successors)
          file << "BB" << t << " -> " << "BB"
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
  std::unordered_map<Node, std::unordered_set<Node>> dt;

  // 1. Sets all nodes dominators to TOP

  std::unordered_set<Node> top;
  goto_cfg::foreach_bb(start, [&top](const Node &n) { top.insert(n); });

  goto_cfg::foreach_bb(start, [&dt, &top](const Node &n) { dt[n] = top; });

  /*
    Dom(n) = {n}, if n = start
             {n} `union` (intersec_pred Dom(p))
   */
  dt[start] = std::unordered_set<Node>({start});

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
    std::unordered_set<Node> intersection = top;
    for (const auto &pred : dominator_node->predecessors)
    {
      assert(dt.count(pred));
      for (const auto &item : top)
        if (!dt[pred].count(item))
          intersection.erase(item);
    }

    // Union of node with its predecessor intersection
    std::unordered_set<Node> result({dominator_node});
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
    log_status("Node");
    node->begin->dump();

    log_status("Dominated");
    for (const auto &edge : edges)
      edge->begin->dump();
  }
}

Dominator::Node Dominator::idom(const Node &n) const
{
  std::vector<Node> sdoms;
  const auto &n_dominators = dom(n);
  std::copy_if(
    n_dominators.begin(),
    n_dominators.end(),
    std::back_inserter(sdoms),
    [&n, this](const auto &item) { return sdom(item, n); });

  for (const Node &n0 : sdoms)
  {
    bool valid_result = true;
    for (const Node &n1 : sdoms)
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
    log_status("Node: {}", key->uuid);
    log_status("Edges");
    for (const auto &n : value)
      log_status("\t{}", n->uuid);
  }
}

std::unordered_set<Dominator::Node> Dominator::dom_frontier(const Node &n) const
{
  assert(dj);
  std::unordered_set<Dominator::Node> result;
  const auto levels = dj->tree.get_levels();
  for (const Node &y : dj->tree.get_subtree(n))
  {
    for (const Node &z : dj->_jedges[y])
      if (levels.at(z) <= levels.at(n))
        result.insert(z);
  }
  return result;
}

std::unordered_set<Dominator::Node>
Dominator::dom_frontier(const std::unordered_set<Dominator::Node> &nodes) const
{
  assert(nodes.size() > 0);

  std::unordered_set<Dominator::Node> result;

  for (const Node &n : nodes)
    for (const Node &df : dom_frontier(n))
      result.insert(df);

  return result;
}

std::unordered_set<Dominator::Node>
Dominator::iterated_dom_frontier(const std::unordered_set<Node> &nodes) const
{
  assert(dj);
  assert(nodes.size() > 0);
  // TODO: There is a linear solution by SREEDHAR using PiggyBanks
  std::unordered_set<Dominator::Node> result = dom_frontier(nodes);
  std::unordered_set<Dominator::Node> result2 = dom_frontier(result);

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

Dominator::DJGraph::DJGraph(const Dominator::Node &cfg, const Dominator &dom)
  : tree(dom)
{
  // A DJ-Graph is a graph composed by D-Edges and J-Edges
  _graph = tree.edges;
  _dedges = tree.edges;
  auto func = [this, &dom](const Node &x)
  {
    auto [val, ins] = _graph.insert({x, std::unordered_set<Node>()});
    for (const Node &y : x->successors)
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
    log_status("Node {}", k->uuid);
    for (const auto &e : edges)
      log_status("\t->{}", e->uuid);
  }
}

std::unordered_map<Dominator::Node, size_t>
Dominator::DomTree::get_levels() const
{
  std::unordered_map<Dominator::Node, size_t> levels;
  levels[root] = 0;

  std::set<Node> worklist({root});
  while (!worklist.empty())
  {
    Node current = *worklist.begin();
    worklist.erase(current);
    if (!edges.count(current))
      continue;

    size_t level = levels[current] + 1;
    for (const Node &nodes : edges.at(current))
    {
      levels[nodes] = level;
      worklist.insert(nodes);
    }
  }
  return levels;
}

std::unordered_set<Dominator::Node>
Dominator::DomTree::get_subtree(const Node &n) const
{
  std::unordered_set<Dominator::Node> nodes({n});
  std::set<Dominator::Node> worklist({n});

  while (!worklist.empty())
  {
    const Node current = *worklist.begin();
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
  goto_cfg::foreach_bb(
    dom.start,
    [this, &dom](const Node &n)
    {
      if (n == root)
        return;

      Node root = dom.idom(n);
      auto [val, ins] = edges.insert({root, std::unordered_set<Node>()});
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

void replace_symbols_in_expr(expr2tc &use, symbolt* symbol,  unsigned last_id_in_bb, const irep_idt var_name)
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

  use->Foreach_operand([symbol, last_id_in_bb, var_name](expr2tc & e)
  {
    replace_symbols_in_expr(e, symbol, last_id_in_bb, var_name);
  }); 
  
}


void ssa_promotion::promote_node(goto_programt &P,const Node &n)
{
  // Assumption: all declarations initialized through an assignment (which can be nondet)
  using Instruction = std::shared_ptr<goto_programt::instructiont>;
  std::unordered_set<std::string> variables;
  std::unordered_map<std::string, std::unordered_set<Node>> defBlocks;
  std::unordered_map<std::string, std::unordered_set<Instruction>>
    defInstructions;

  std::unordered_map<std::string, std::unordered_set<Node>> useBlocks;
  std::unordered_map<std::string, std::unordered_set<Instruction>>
    useInstructions;

  auto insert_definition = [&variables, &defBlocks, &defInstructions](
                             const irep_idt &var,
                             goto_programt::instructionst::iterator &I,
                             const Node &bb)
  {
    const std::string var_name = var.as_string();
    variables.insert(var_name);

    // TODO: C++20 anonymous var for "ins" :)
    auto [inst, ins1] =
      defInstructions.insert({var_name, std::unordered_set<Instruction>()});
    auto [block, ins2] =
      defBlocks.insert({var_name, std::unordered_set<Node>()});

    defInstructions[var_name].insert(
      std::make_shared<goto_programt::instructiont>(*I));
    defBlocks[var_name].insert(bb);
  };

  auto insert_use = [&variables, &useBlocks, &useInstructions, &defBlocks](
                      const expr2tc &e,
                      goto_programt::instructionst::iterator &I,
                      const Node &bb)
  {
    std::unordered_set<expr2tc, irep2_hash> symbols;
    get_symbols(e, symbols);
    for (const auto &var : symbols)
    {
      assert(is_symbol2t(var));
      const std::string var_name = to_symbol2t(var).thename.as_string();
      variables.insert(var_name);

      // TODO: C++20 anonymous var for "ins" :)
      auto [inst, ins1] =
        useInstructions.insert({var_name, std::unordered_set<Instruction>()});
      auto [block, ins2] =
        useBlocks.insert({var_name, std::unordered_set<Node>()});

      useInstructions[var_name].insert(
        std::make_shared<goto_programt::instructiont>(*I));

      // Only consider a use block if the value is used before redefined
      if(!defBlocks[var_name].count(bb))
        useBlocks[var_name].insert(bb);
    }
  };

  goto_cfg::foreach_bb(
    n,
    [&insert_definition, &insert_use](const Node &bb)
    {
      for (goto_programt::instructionst::iterator start = bb->begin;
           start != bb->end;
           start++)
      {
        if (start->is_decl());
          //insert_definition(to_code_decl2t(start->code).value, start, bb);
        else if (start->is_assign())
        {
          const expr2tc &target = to_code_assign2t(start->code).target;
          if (is_symbol2t(target))
            insert_definition(to_symbol2t(target).thename, start, bb);

          insert_use(to_code_assign2t(start->code).source, start, bb);
        }
        else if (start->is_function_call())
          for (const auto &op : to_code_function_call2t(start->code).operands)
            insert_use(op, start, bb);
        else
        {
          insert_use(start->code, start, bb);
          insert_use(start->guard, start, bb);
        }
      }
    });

  // Compute phi-nodes (dominance frontier of all nodes that defines the variable)
  Dominator info(n);
  auto ptr = std::make_shared<Dominator::DJGraph>(n, info);
  auto dj_graph = *ptr;
  info.dj = ptr; 
  
  for (auto &var : variables)
  {
    const type2tc symbol_type = int_type2(); // todo: get symbol
    
    // Some symbols might come from globals or functions
    if (defBlocks[var].empty()) 
      continue;

    const auto phinodes = info.dom_frontier(defBlocks[var]);

    // For each definition, create a symbol:
    size_t symbol_counter = defInstructions[var].size() + phinodes.size();

    // Lets add all symbols at the beginning
    std::vector<symbolt *> symbols;
    std::unordered_map<std::string, unsigned> symbol_to_index;
    for (int i = 0; i < symbol_counter; i++)
    {
      symbolt symbol;
      symbol.type = migrate_type_back(symbol_type);
      symbol.id = fmt::format("{}.{}", var, i);
      symbol.name = symbol.id;
      symbol.mode = "C"; // todo: get mode
      symbol.location = n->begin->location;

      symbolt *symbol_in_context = context.move_symbol_to_context(symbol);
      assert(symbol_in_context != nullptr);
      symbols.push_back(symbol_in_context);
      symbol_to_index[symbol_in_context->id.as_string()] = i;

      
      goto_programt::instructiont decl;
      decl.make_decl();
      decl.code = code_decl2tc(symbol_type, symbol_in_context->id);
      decl.location = n->begin->location;
      P.insert_swap(n->begin, decl);
      
    }

    unsigned counter = 0;
    std::unordered_map<unsigned, Node> ref_to_nodes; // the cfg locations are no longer reliable
    // Insert phi-nodes
    for (const Node& phinode : phinodes)
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
      auto phi_symbol = symbol2tc(symbol_type, symbols[counter++]->id);

      const expr2tc lhs = symbol2tc(symbol_type, symbols[0]->id);
      const expr2tc rhs = symbol2tc(symbol_type, symbols[0]->id);
      auto pred_it = phinode->predecessors.begin();
      const unsigned lhs_location = (*pred_it)->begin->location_number;
      ref_to_nodes[lhs_location] = *pred_it;
      pred_it++;

      const unsigned rhs_location = (*pred_it)->begin->location_number;
      ref_to_nodes[rhs_location] = *pred_it;
      const expr2tc phi_expr =
        phi2tc(symbol_type, lhs, rhs, lhs_location, rhs_location);
      phi_instr.code = code_assign2tc(phi_symbol, phi_expr);

      P.insert_swap(phinode->begin, phi_instr);
      defBlocks[var].insert(phinode);
      useBlocks[var].insert(phinode);
    }


    std::unordered_map<Node, unsigned> last_id_in_bb;


    // For each definition rename the symbol.
    for (auto defBlock : defBlocks[var])
    {
      for (auto instruction = defBlock->begin; instruction != defBlock->end; instruction++)
      {
        if (!instruction->is_assign())
          continue;

        const auto code_expr = to_code_assign2t(instruction->code);
        if (!(is_symbol2t(code_expr.target) &&
              to_symbol2t(code_expr.target).thename == var))
          continue;
        // TODO: assert(code_expr.target->type == symbol_type);

        last_id_in_bb[defBlock] = counter;
        auto new_symbol = symbol2tc(symbol_type, symbols[counter++]->id);
        instruction->code = code_assign2tc(new_symbol, code_expr.source);        
      }
    }

    auto get_last_defined = [&last_id_in_bb](const Node n) -> unsigned
    {
      if(!last_id_in_bb.count(n))
        return 0;
      return last_id_in_bb[n];
    };

    auto find_location = [&get_last_defined, &ref_to_nodes, &symbols](const Node start, unsigned id) -> symbolt *
    {
      // Explore in reverse order:
      std::vector<Node> worklist{start};
      return symbols[get_last_defined(ref_to_nodes[id])];
    };

    // For each use rename the symbol
    for (auto useBlock : useBlocks[var])
    {
      unsigned current_last_id = 0;
      for (auto instruction = useBlock->begin; instruction != useBlock->end; instruction++)
      {

        if (
          instruction->is_decl() &&
          to_code_decl2t(instruction->code).value == var)
           instruction->make_skip();
        else if (instruction->is_assign())
        {
          auto assign = to_code_assign2t(instruction->code);
          assert(current_last_id <= symbols.size());

          if(!is_phi2t(assign.source))
            replace_symbols_in_expr(
              assign.source, symbols[current_last_id], 0, var);
          else
          {
            // TODO set location correctly
            auto lhs_symbol = find_location(useBlock, to_phi2t(assign.source).lhs_location);
            auto rhs_symbol =
              find_location(useBlock, to_phi2t(assign.source).rhs_location);

            if (!lhs_symbol || !rhs_symbol)
              continue;
          
            to_phi2t(assign.source).lhs = symbol2tc(
              symbol_type,
              lhs_symbol->id);
            to_phi2t(assign.source).rhs = symbol2tc(
              symbol_type,
              rhs_symbol->id);
            
          }

          if (is_symbol2t(assign.target) &&
                symbol_to_index.count(to_symbol2t(assign.target).thename.as_string()))
            current_last_id = symbol_to_index.at(
              to_symbol2t(assign.target).thename.as_string());

          instruction->code = code_assign2tc(assign.target, assign.source);
        }         
 
      }
    }
    // Kill all symbols at the end ?
  }
}

