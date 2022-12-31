#pragma once

#include <util/message.h>
#include <goto-programs/goto_functions.h>
#include <unordered_set>
#include <string>

/** Dirty variables are ones which have their address taken so we can't
 * reliably work out where they may be assigned and are also considered shared
 * state in the presence of multi-threading. */
class dirtyt
{
public:
  dirtyt() = default;

  explicit dirtyt(const goto_functiont &goto_function)
  {
    build(goto_function);
  }

  explicit dirtyt(const goto_functionst &goto_functions)
  {
    build(goto_functions);
  }

  void output(std::ostream &out) const;

  bool operator()(const std::string &id) const
  {
    log_debug("[dirty] Looking for {}", id);
    return dirty.find(id) != dirty.end();
  }

  bool operator()(const symbol2t &expr) const
  {
    return operator()(expr.thename.c_str());
  }

  const std::unordered_set<std::string> &get_dirty_ids() const
  {
    return dirty;
  }

  void add_function(const goto_functiont &goto_function)
  {
    build(goto_function);
  }

  void build(const goto_functionst &goto_functions)
  {
    for(const auto &gf_entry : goto_functions.function_map)
      build(gf_entry.second);
  }

protected:
  void build(const goto_functiont &goto_function);

  // variables whose address is taken
  std::unordered_set<std::string> dirty;
  void search_other(const goto_programt::instructiont &instruction);
  void find_dirty(const expr2tc &expr);
  void find_dirty_address_of(const expr2tc &expr);
};

inline std::ostream &operator<<(std::ostream &out, const dirtyt &dirty)
{
  dirty.output(out);
  return out;
}

/// Wrapper for dirtyt that permits incremental population, ensuring each
/// function is analysed exactly once.
class incremental_dirtyt
{
public:
  /**
 * Analyse the given function with dirtyt if it hasn't been seen before
 * @param id function id to analyse
 * @param function function to analyse
 */
  void populate_dirty_for_function(
    const std::string &id,
    const goto_functiont &function);

  bool operator()(const std::string &id) const
  {
    return dirty(id);
  }

  bool operator()(const symbol2t &expr) const
  {
    return dirty(expr);
  }

private:
  dirtyt dirty;
  std::unordered_set<std::string> dirty_processed_functions;
};
