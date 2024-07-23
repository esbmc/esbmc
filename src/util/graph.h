#pragma once

#include <util/message.h>

template <class V>
class graph
{
public:
  graph() = default;

  void add_vertex(const V &);
  void add_edge(const V &, const V &);

  std::vector<V>
  dfs(const V &start, std::function<void(const V &)> lambda) const;

  std::unordered_set<V> vertexes() const;
  std::unordered_set<std::pair<V,V>> edges() const;

  void dump() const;

private:
  void acc_dfs(const V &start, std::vector<V> &acc, std::unordered_set<V> &visited, std::function<void(const V&)> lambda) const;
  using _Graph = std::unordered_map<V, std::set<V>>;
  _Graph _g;
};

template <class V>
void graph<V>::add_vertex(const V &vertex)
{
  // Add vertex
  _g.insert({vertex, std::set<V>()});
}

template <class V>
void graph<V>::add_edge(const V &v1, const V &v2)
{
  // Add edge
  _g[v1].insert(v2);
}

template <class V>
void graph<V>::dump() const
{
  // Dump graph
  log_status("Dumping graph");
  for (const auto &[v0, edges] : _g)
  {
    log_status("Node: {}", v0);
    for (const V &v1 : edges)
      log_status("-> {}", v1);
  }
}

template <class V>
std::vector<V> graph<V>::dfs(const V &start, std::function<void(const V&)> lambda) const
{
  std::vector<V> acc;
  std::unordered_set<V> visited;
  acc_dfs(start, acc, visited, lambda);
  return acc;
}


template <class V>
void graph<V>::acc_dfs(const V &start, std::vector<V> &acc, std::unordered_set<V> &visited, std::function<void(const V&)> lambda) const
{
  auto [it,ins] = visited.emplace(start);
  if (!ins)
    return;
  acc.push_back(start);
  lambda(start);
  for (const auto &v : _g.at(start))
    acc_dfs(v, acc, visited, lambda);  
}
