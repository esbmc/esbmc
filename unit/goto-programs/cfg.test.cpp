#include <memory>
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/goto_cfg.h>

const mode_table_et mode_table[] = {
};

using IntGraph = std::unordered_map<int,std::unordered_set<int>>;
std::shared_ptr<goto_cfg::basic_block> from_int_graph(IntGraph graph, int start, int end)
{
  std::map<int, std::shared_ptr<goto_cfg::basic_block>> blocks;
  for (int i = start; i <= end; i++)
  {
    blocks[i] = std::make_shared<goto_cfg::basic_block>();
    blocks[i]->uuid = i;
  }
  for (int i = start; i <= end; i++)
  {
    for (auto &suc : graph[i])
    {

      blocks[i]->successors.insert(blocks[suc]);
      blocks[suc]->predecessors.insert(blocks[i]);
    }
  }
  return blocks[start];
}


TEST_CASE(
  "DJ-Graphs",
  "[cfg][ssa]")
{
  // SREEDHAR, Vugranam C.; GAO, Guang R. Computing phi-nodes in linear time using DJ graphs. Journal of Programming Languages, v. 3, p. 191-214, 1995.
  // See page 7 for nice pictures

  IntGraph graph;
  graph[0] = std::unordered_set<int>({1, 16}); // START
  graph[1] = std::unordered_set<int>({4, 2, 3});
  graph[2] = std::unordered_set<int>({4, 7});
  graph[3] = std::unordered_set<int>({9});
  graph[4] = std::unordered_set<int>({5});
  graph[5] = std::unordered_set<int>({6});
  graph[6] = std::unordered_set<int>({8, 2});
  graph[7] = std::unordered_set<int>({8});
  graph[8] = std::unordered_set<int>({15, 7});
  graph[9] = std::unordered_set<int>({10, 11});
  graph[10] = std::unordered_set<int>({12});
  graph[11] = std::unordered_set<int>({12});
  graph[12] = std::unordered_set<int>({13});
  graph[13] = std::unordered_set<int>({14, 3, 15});
  graph[14] = std::unordered_set<int>({12});
  graph[15] = std::unordered_set<int>({16});
  graph[16] = std::unordered_set<int>({}); // END

  std::shared_ptr<goto_cfg::basic_block> bb = from_int_graph(graph, 0, 16);
  Dominator info(bb);
  const Dominator::DomTree dt(info);
  SECTION("Dominator Tree")
  {        
    REQUIRE(dt.root->uuid == 0);

    const std::array expected{
      std::unordered_set<int>({1, 16}),
      std::unordered_set<int>({2, 4, 7, 8, 15, 3}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({9}),
      std::unordered_set<int>({5}),
      std::unordered_set<int>({6}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({10, 11, 12}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({13}),
      std::unordered_set<int>({14}),
      std::unordered_set<int>({}),
      std::unordered_set<int>({})};

    const std::unordered_set<int> nodes{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const std::unordered_set<int> leaves{2, 6, 7,8,15, 14, 10, 11};

    std::unordered_set<int> visited;    
    for (auto [k, v] : dt.edges)
    {
      CAPTURE(k->uuid);
      visited.insert(k->uuid);
      std::unordered_set<int> actual;
      REQUIRE(!leaves.count(k->uuid));
      for (auto &e : v)
      {
        visited.insert(e->uuid);
        actual.insert(e->uuid);
      }
      REQUIRE(actual == expected[k->uuid]);
    }

    REQUIRE(visited == nodes);
  }

  SECTION("Dominator Tree Levels")
  {
    auto levels = dt.get_levels();
    const std::array expected{0, 1, 2, 2, 2, 3, 4, 2, 2, 3, 4, 4, 4, 5, 6, 2, 1};

    for (auto [k, v] : dt.edges)
    {
      CAPTURE(k->uuid);
      REQUIRE((size_t)expected[k->uuid] == levels[k]);
      for (const auto &node : v)
      {
        CAPTURE(node->uuid);
        REQUIRE((size_t)expected[node->uuid] == levels[node]);
      }
    }
  }

  SECTION("Dominator Tree Subtrees")
  {
    std::shared_ptr<goto_cfg::basic_block> node;

    for (auto [k, v] : dt.edges)
    {
      if (k->uuid == 3)
      {
        node = k;
        break;
      }
    }

    auto subtree = dt.get_subtree(node);
    const std::unordered_set expected{3, 9, 10, 11, 12, 13, 14};

    REQUIRE(subtree.size() == expected.size());

    for (const auto &s : subtree)
      REQUIRE(expected.count(s->uuid));
  }


  auto ptr = std::make_shared<Dominator::DJGraph>(dt, bb, info);
  auto dj_graph = *ptr;
  info.dj = ptr;
  SECTION("DJ-GRAPHS")
  {
    const std::array expected{
      std::unordered_set<int>({1, 16}),
        std::unordered_set<int>({2, 4, 7, 8, 15, 3}),
        std::unordered_set<int>({7, 4}), std::unordered_set<int>({9}),
        std::unordered_set<int>({5}), std::unordered_set<int>({6}),
        std::unordered_set<int>({2, 8}), std::unordered_set<int>({8}),
        std::unordered_set<int>({7, 15}), std::unordered_set<int>({10, 11, 12}),
        std::unordered_set<int>({12}), std::unordered_set<int>({12}),
        std::unordered_set<int>({13}), std::unordered_set<int>({14, 15, 3}),
        std::unordered_set<int>({12}), std::unordered_set<int>({16}),
    std::unordered_set<int>({})};

    std::unordered_set<int> visited;    
    for (auto [k, v] : dj_graph._graph)
    {
      CAPTURE(k->uuid);
      visited.insert(k->uuid);
      std::unordered_set<int> actual;
      for (auto &e : v)
      {
        visited.insert(e->uuid);
        actual.insert(e->uuid);
      }
      REQUIRE(actual == expected[k->uuid]);
    }
    const std::unordered_set<int> nodes{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    REQUIRE(nodes == visited);
  }

  SECTION("Dominance Frontier (Node)")
  {
    std::shared_ptr<goto_cfg::basic_block> node;

    for (auto [k, v] : dt.edges)
    {
      if (k->uuid == 3)
      {
        node = k;
        break;
      }
    }

    auto frontier = info.dom_frontier(node);
    const std::unordered_set<size_t> expected {3,15};
    REQUIRE(frontier.size() == 2);
    for (auto &df : frontier)
      REQUIRE(expected.count(df->uuid));
  }

  SECTION("Dominance Frontier (Node 2)")
  {
    std::shared_ptr<goto_cfg::basic_block> node;

    for (auto [k, v] : dt.edges)
    {
      if (k->uuid == 9)
      {
        node = k;
        break;
      }
    }

    auto frontier = info.dom_frontier(node);
    const std::unordered_set<size_t> expected {3,15};
    REQUIRE(frontier.size() == 2);
    for (auto &df : frontier)
      REQUIRE(expected.count(df->uuid));
  }

  SECTION("Dominance Frontier (Set)")
  {
    std::unordered_set<std::shared_ptr<goto_cfg::basic_block>> nodes;

    for (auto [k, v] : dt.edges)
    {
      switch (k->uuid)
      {
      case 3:
      case 9:
        nodes.insert(k);
        break;
      default:
        break;
      }
    }

    auto frontier = info.dom_frontier(nodes);
    const std::unordered_set expected{3, 15};

    REQUIRE(frontier.size() == expected.size());
    
    for (auto &df : frontier)
    {
      CAPTURE(df->uuid);
      REQUIRE(expected.count(df->uuid));    
    }
      
  }

  SECTION("Iterated Dominance Frontier (set)")
  {
    std::unordered_set<std::shared_ptr<goto_cfg::basic_block>> nodes;

    for (auto [k, v] : dt.edges)
    {
      switch (k->uuid)
      {
      case 0:
      case 5:
      case 13:
        nodes.insert(k);
        break;
      default:
        break;
      }
    }

    auto frontier = info.iterated_dom_frontier(nodes);
    const std::unordered_set expected {2,3,4,7,8,12,15,16}; 
    for (auto &df : frontier)
      REQUIRE(expected.count(df->uuid));
   }
 }
