#pragma once

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

/**
   * @brieft Dominator class to compute all dominator info
   *
   * The dominator information is useful to compute featues over the CFG.
   * For ESBMC, the main purpose here is to be able to compute a DJ-Graph
   * which can be used for SSA-promotion and loop invariants.
   */
struct Dominator
{
  using Node = std::shared_ptr<goto_cfg::basic_block>;

  /// First node in the CFG
  const Node &start;

  /**
     * @brief Dominator Tree has the property that a parent dominates its
     * children
    */
  struct DomTree
  {
    DomTree(const Dominator &dom);
    const Node &root;
    std::unordered_map<Node, std::unordered_set<Node>> edges;

    std::unordered_map<Node, size_t> get_levels() const;
    std::unordered_set<Node> get_subtree(const Node &n) const;
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
    DJGraph(const Dominator::Node &cfg, const Dominator &dom);

    using Graph = std::unordered_map<Node, std::unordered_set<Node>>;

    /// J-Edges are x->y edges from the CFG such that x !sdom y. y is called join node
    Graph _jedges;
    /// D-Edges are the edges from the dominator tree
    Graph _dedges;
    /// The full graph is D-Edges union J-Edges
    Graph _graph;

    void dump() const;
  };

  Dominator(const Node &start) : start(start)
  {
    compute_dominators();
  }

  // Evaluates whether n1 dom n2
  inline bool dom(const Node &n1, const Node &n2) const
  {
    return dom(n2).count(n1);
  }

  inline bool sdom(const Node &n1, const Node &n2) const
  {
    return n1 != n2 && dom(n1, n2);
  }

  // Returns the immediate dominator of n.  The idom of a
  // node n1 is the unique node n2 that n2 sdom n1 but does not sdom any other node that sdom n1.
  Node idom(const Node &n) const;

  void dump_dominators() const;
  void dump_idoms() const;

  std::unordered_set<Node> dom_frontier(const Node &n) const;
  std::unordered_set<Node>
  dom_frontier(const std::unordered_set<Node> &nodes) const;
  std::unordered_set<Node>
  iterated_dom_frontier(const std::unordered_set<Node> &n) const;

  std::shared_ptr<DJGraph> dj;

private:
  void compute_dominators();
  std::unordered_map<Node, std::unordered_set<Node>> _dominators;
  // Get dominators of a node
  inline std::unordered_set<Node> dom(const Node &node) const
  {
    return _dominators.at(node);
  }
};

class live_analysis
{
  using Node = std::shared_ptr<goto_cfg::basic_block>;

public:
  live_analysis(const Node &root);
  inline std::unordered_set<Node> get_live_blocks() const
  {
    return _live_blocks;
  }

private:
  void compute_blocks();
  std::unordered_set<Node> _live_blocks;
};

class ssa_promotion
{
  using Node = std::shared_ptr<goto_cfg::basic_block>;
public:
  ssa_promotion(goto_cfg &cfg) : cfg(cfg) {}

  void promote();

protected:
  void promote_node(const Node& n);
  void insert_phi(const live_analysis &var);
  void rename_phi(const live_analysis &var);

private:
  goto_cfg &cfg;
  const std::unordered_set<std::string> _skip{"__ESBMC_main"};
};
