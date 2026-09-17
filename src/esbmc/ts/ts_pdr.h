#pragma once

#include <esbmc/ts/ts_engines.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

/// IC3/PDR over bit-level cubes of the transition system's state. Frames are
/// delta-encoded clause sets, each level enabled by an activation literal, so
/// every query runs in the one solver the base engine set up.
class ts_pdrt : public ts_enginet
{
public:
  ts_pdrt(
    const transition_systemt &ts,
    const namespacet &ns,
    optionst &options);

  enum class resultt
  {
    safe,
    unsafe,
    unknown
  };

  /// On `unsafe`, `cex_depth` is the step at which BMC finds the violation.
  resultt run(uint64_t max_frames, unsigned &cex_depth);

  /// Why run() returned `unknown`.
  std::string reason;

private:
  struct litt
  {
    unsigned bit;
    bool value;
    bool operator<(const litt &o) const
    {
      return bit < o.bit || (bit == o.bit && value < o.value);
    }
    bool operator==(const litt &o) const
    {
      return bit == o.bit && value == o.value;
    }
  };
  /// Sorted by bit.
  using cubet = std::vector<litt>;

  struct obligationt
  {
    unsigned level;
    cubet cube;
    unsigned steps_to_bad;
  };

  optionst *options_;
  /// C(s,i): the step's assumptions, back-edge guard and !Bad at step 0.
  expr2tc trans_expr, bad_expr;
  /// Per bit, the initial value when the component starts at a constant.
  std::vector<int> init_bit; // -1 unknown, 0, 1
  std::map<std::string, std::pair<unsigned, double>> query_stats;
  std::vector<expr2tc> component_cur;
  std::vector<std::pair<unsigned, unsigned>> bit_of; // (component, bit)
  std::vector<expr2tc> cur_bit, next_bit;             // "bit is 1"
  std::vector<std::vector<cubet>> frames;             // frames[level]
  std::vector<expr2tc> level_lits;
  expr2tc act_trans;
  unsigned queries = 0;

  bool build_bits();
  expr2tc literal(const litt &l, bool next) const;
  expr2tc cube_expr(const cubet &c, bool next) const;
  std::vector<expr2tc> frame(unsigned level) const;
  smt_resultt
  solve(const std::vector<expr2tc> &assumptions, const std::string &kind);
  cubet model_cube();
  /// Shrink the state cube `s` of the last SAT model to the bits that, with
  /// the same inputs, still force `target`.
  cubet lift(const cubet &s, const expr2tc &target);
  bool intersects_init(const cubet &c);
  bool subsumed(const cubet &c, unsigned level) const;
  void add_blocked(const cubet &c, unsigned level);
  /// F[level-1] & !c & T & c' is UNSAT; on success `core` holds the literals
  /// of c the proof used.
  bool relative_inductive(const cubet &c, unsigned level, cubet &core);
  cubet generalize(const cubet &c, unsigned level, const cubet &core);
  bool block(obligationt bad, unsigned &cex_depth);
  /// Returns the first level whose delta emptied, or 0.
  unsigned propagate();
  bool check_invariant(unsigned from_level);
};
