#pragma once

#include "context.h"
#include <optional>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string_view>
#include <goto-programs/goto_program.h>
#include <goto-programs/goto_functions.h>

/**
 * @brief An implementation of a control flow graph for goto programs.
 *
 * This class manipulates and transform a goto program by using a CFG abstraction.
 */
class goto_cfg
{
public:
  goto_cfg(goto_functionst &goto_functions);

  /**
   * @brief Generates a dot file containing the CFG.
   *
   * @param filename output file name
   */
  void dump_graph() const;

  /**
   * @brief A basic block is a sequence of instructions that has no branches in it.
   *
   * It consists of a sequence of instructions until a leader is found.
   * A leader consists in operations that create a new basic block,
   * i.e., label, if-goto, return, throw, catch, etc.
   */
  struct basic_block
  {
    enum class terminator_type
    {
      OTHER,
      IF_GOTO
    };
    goto_programt::instructionst::iterator begin;
    goto_programt::instructionst::iterator end;
    std::set<std::shared_ptr<basic_block>> successors;
    std::unordered_set<std::shared_ptr<basic_block>> predecessors;
    terminator_type terminator = terminator_type::OTHER;
    int uuid;

    template <class F>
    void foreach_inst(F);

    template <class F>
    void foreach_bb(F);
  };

  std::unordered_map<std::string, std::vector<std::shared_ptr<basic_block>>>
    basic_blocks;

  template <class F>
  static void foreach_bb(const std::shared_ptr<basic_block> &start, F);
};

using CFGNode = std::shared_ptr<goto_cfg::basic_block>;

/**
   * @brieft Dominator class to compute all dominator info
   *
   * The dominator information is useful to compute featues over the CFG.
   * For ESBMC, the main purpose here is to be able to compute a DJ-Graph
   * which can be used for SSA-promotion and loop invariants.
   */
struct Dominator
{
  /// First node in the CFG
  const CFGNode &start;

  /**
     * @brief Dominator Tree has the property that a parent dominates its
     * children
    */
  struct DomTree
  {
    DomTree(const Dominator &dom);
    const CFGNode &root;
    std::unordered_map<CFGNode, std::unordered_set<CFGNode>> edges;

    std::unordered_map<CFGNode, size_t> get_levels() const;
    std::unordered_set<CFGNode> get_subtree(const CFGNode &n) const;
  };

  /**
     * @brief DJ-Graph is a graph used to improve the computability of
     * the domain frontier of the CFG.
     *
     * See SREEDHAR, Vugranam C.; GAO, Guang R. Computing phi-nodes in linear time using DJ graphs. Journal of Programming Languages, v. 3, p. 191-214, 1995.
     */
  struct DJGraph
  {
    const DomTree tree;
    DJGraph(const CFGNode &cfg, const Dominator &dom);

    using Graph = std::unordered_map<CFGNode, std::unordered_set<CFGNode>>;

    /// J-Edges are x->y edges from the CFG such that x !sdom y. y is called join node
    Graph _jedges;
    /// D-Edges are the edges from the dominator tree
    Graph _dedges;
    /// The full graph is D-Edges union J-Edges
    Graph _graph;

    void dump() const;
  };

  Dominator(const CFGNode &start) : start(start)
  {
    compute_dominators();
    dj = std::make_shared<Dominator::DJGraph>(start, *this); 
  }

  // Evaluates whether n1 dom n2
  inline bool dom(const CFGNode &n1, const CFGNode &n2) const
  {
    return dom(n2).count(n1);
  }

  inline bool sdom(const CFGNode &n1, const CFGNode &n2) const
  {
    return n1 != n2 && dom(n1, n2);
  }

  // Returns the immediate dominator of n.  The idom of a
  // node n1 is the unique node n2 that n2 sdom n1 but does not sdom any other node that sdom n1.
  CFGNode idom(const CFGNode &n) const;

  void dump_dominators() const;
  void dump_idoms() const;

  std::unordered_set<CFGNode> dom_frontier(const CFGNode &n) const;
  std::unordered_set<CFGNode>
  dom_frontier(const std::unordered_set<CFGNode> &nodes) const;
  std::unordered_set<CFGNode>
  iterated_dom_frontier(const std::unordered_set<CFGNode> &n) const;

  std::shared_ptr<DJGraph> dj;

private:
  void compute_dominators();
  std::unordered_map<CFGNode, std::unordered_set<CFGNode>> _dominators;
  // Get dominators of a node
  inline std::unordered_set<CFGNode> dom(const CFGNode &node) const
  {
    return _dominators.at(node);
  }
};

class ssa_promotion
{
public:
  ssa_promotion(goto_cfg &cfg, goto_functionst &goto_functions, contextt &context) : cfg(cfg), goto_functions(goto_functions), context(context)
  {
  }

  void promote();

protected:
  void promote_node(goto_programt &P, const CFGNode &n);

private:
  std::unordered_set<std::string> collect_symbols();
  goto_cfg &cfg;
  contextt &context;
  goto_functionst &goto_functions;
  const std::unordered_set<std::string> _skip{"__ESBMC_main", "__ESBMC_pthread_start_main_hook","__ESBMC_pthread_end_main_hook", "c:@F@__ESBMC_atexit_handler"};
};
