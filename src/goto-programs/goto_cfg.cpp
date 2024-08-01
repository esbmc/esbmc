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
    promote_node(v[0]);
  }
}


#if 0
  // Compute live analysis blocks for each variable
  using VarDomain = std::string;

  // The set of variables that are used in s before any assignment in the same basic block.
  gen_kill<VarDomain>::DataflowSet gen;
  // The set of variables that are assigned a value in s
  gen_kill<VarDomain>::DataflowSet kill;

  auto dataflow_init =
    [](
      gen_kill<VarDomain>::DataflowSet &set,
      std::unordered_map<std::string, std::unordered_set<Node>> input)
  {
    assert(set.empty());
    for (auto &[k, v] : input)
    {
      for (const auto &n : v)
      {
        auto [val, ins] = set.insert({n, std::set<VarDomain>()});
        val->second.insert(k);
      }
    }
  };

  dataflow_init(gen, useBlocks);
  dataflow_init(kill, defBlocks);

  gen_kill<VarDomain> live_analysis(n, gen, kill, false);
#endif

void ssa_promotion::promote_node(const Node &n)
{
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
        if (start->is_decl())
          insert_definition(to_code_decl2t(start->code).value, start, bb);
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

  // Compute phi-nodes

  // TODO: use the f live_analysis :)

  // The phi-nodes are the dominance frontier of the
  // of the blocks that defines a var

  Dominator info(n);
  auto ptr = std::make_shared<Dominator::DJGraph>(n, info);
  auto dj_graph = *ptr;
  info.dj = ptr; 
  
  for (auto &var : variables)
  {    
    if (defBlocks[var].empty())
      continue;

    const auto phinodes = info.dom_frontier(defBlocks[var]);
    // Insert phi-node

    // rename phi-nodes
  }
}

template <class Domain>
gen_kill<Domain>::gen_kill(
  const Node &bb,
  const DataflowSet &gen,
  const DataflowSet &kill,
  bool forward,
  bool confluence_is_union)
  : forward(forward), confluence_is_union(confluence_is_union)
{
  if (forward || !confluence_is_union)
  {
    log_error("Gen Kill does not support this configuration yet");
    abort();
  }
  std::set<Node> worklist;
  goto_cfg::foreach_bb(
    bb,
    [&worklist, this](const Node &n)
    {
      worklist.insert(n);
      in.insert({n, std::set<Domain>()});
      out.insert({n, std::set<Domain>()});

    });

  if (!worklist.size())
  {
    log_warning("Something strange is happening with current CFG");
    bb->begin->dump();
    return;
  }

  while (!worklist.empty())
  {
    const Node s = *worklist.begin();
    worklist.erase(s);

    // The domain is a lattice, we can just count the elements to check for diffs
    const size_t in_before = in[s].size();
    const size_t out_before = out[s].size();
    
    for (const auto &pred : s->successors)
    {
      for (const auto &pred_in : in[pred])
        out[s].insert(pred_in);
    }

    if (gen.count(s))
      for (const auto &g : gen.at(s))
        in[s].insert(g);

    for (const auto &l : out[s])
    {
      if (kill.count(s) && kill.at(s).count(l))
        continue;

      in[s].insert(l);
    }

    if (in[s].size() != in_before || out[s].size() != out_before)
      for (const auto &pred : s->predecessors)
        worklist.insert(pred);
  }
}
