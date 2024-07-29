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
  // SREEDHAR, Vugranam C.; GAO, Guang R. Computing -nodes in linear time using DJ graphs. Journal of Programming Languages, v. 3, p. 191-214, 1995.
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
  const goto_cfg::Dominator info(bb);
  const auto dt = info.dom_tree();
  SECTION("Dominator Tree")
  {        
    REQUIRE(dt.first->uuid == 0);

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
    for (auto [k, v] : dt.second)
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

  const goto_cfg::Dominator::DJGraph dj_graph(dt, bb, info);
  SECTION("DJ-GRAPHS")
  {
    const std::array expected{
      std::unordered_set<int>({1, 16}),
      std::unordered_set<int>({2, 4, 7, 8, 15, 3}),
      std::unordered_set<int>({7,4}),
      std::unordered_set<int>({9}),
      std::unordered_set<int>({5}),
      std::unordered_set<int>({6}),
      std::unordered_set<int>({2,8}),
      std::unordered_set<int>({8}),
      std::unordered_set<int>({7,15}),
      std::unordered_set<int>({10, 11, 12}),
      std::unordered_set<int>({12}),
      std::unordered_set<int>({12}),
      std::unordered_set<int>({13}),
      std::unordered_set<int>({14,15,3}),
      std::unordered_set<int>({12}),
      std::unordered_set<int>({16})};

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

  SECTION("phi-Nodes")
  {
    
  }
}
