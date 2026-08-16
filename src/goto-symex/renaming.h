#ifndef _GOTO_SYMEX_RENAMING_H_
#define _GOTO_SYMEX_RENAMING_H_

#include <set>
#include <goto-symex/level1_map.h>
#include <goto-symex/symex_invariant.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2_guard.h>
#include <util/base/i2string.h>
#include <irep2/irep2_expr.h>
#include <util/irep/std_expr.h>

class namespacet;

namespace renaming
{
struct renaming_levelt
{
public:
  virtual void get_original_name(expr2tc &expr) const = 0;
  virtual void rename(expr2tc &expr) = 0;
  virtual void remove(const expr2tc &symbol) = 0;

  virtual void get_ident_name(expr2tc &symbol) const = 0;

  virtual ~renaming_levelt() = default;
  //  protected:
  //  XXX: should leave protected enabled, but g++ 5.4 on ubuntu 16.04 does not
  //  appear to honor the following friend directive?
  static void get_original_name(expr2tc &expr, symbol2t::renaming_level lev);
  friend void build_goto_symex_classes();
};

// level 1 -- function frames
// this is to preserve locality in case of recursion

struct level1t : public renaming_levelt
{
public:
  struct name_rec_hash;
  class name_record
  {
  public:
    name_record() : base_name("")
    {
    }

    name_record(const symbol2t &sym) : base_name(sym.thename)
    {
    }

    name_record(const irep_idt &name) : base_name(name)
    {
    }

    int compare(const name_record &ref) const
    {
      if (base_name.get_no() < ref.base_name.get_no())
        return -1;
      if (base_name.get_no() > ref.base_name.get_no())
        return 1;

      return 0;
    }

    bool operator<(const name_record &ref) const
    {
      if (compare(ref) == -1)
        return true;
      return false;
    }

    bool operator==(const name_record &ref) const
    {
      if (compare(ref) == 0)
        return true;
      return false;
    }

    irep_idt base_name;

    friend struct renaming::level1t::name_rec_hash;
  };

  struct name_rec_hash
  {
    size_t operator()(const name_record &ref) const
    {
      return ref.base_name.get_no();
    }

    bool operator()(const name_record &ref, const name_record &ref2) const
    {
      return ref < ref2;
    }
  };

  typedef persistent_map<name_record, unsigned, name_rec_hash> current_namest;
  current_namest current_names;
  unsigned int thread_id;
  // Set externally (alongside thread_id) so rename() / get_ident_name()
  // can check is_thread_local on globals and route them per-thread
  // instead of to the shared level1_global bucket (issue #4434, #4433).
  const namespacet *ns = nullptr;

  void rename(expr2tc &expr) override;
  void get_ident_name(expr2tc &symbol) const override;
  void remove(const expr2tc &symbol) override
  {
    current_names.erase(name_record(to_symbol2t(symbol)));
  }

  void rename(const expr2tc &symbol, unsigned frame)
  {
    // Given that this is level1, use base symbol.
    name_record rec(to_symbol2t(symbol));
    const unsigned *cur = current_names.find(rec);
    // I1 at L1: an activation index only ever grows.
    SYMEX_INVARIANT(
      !cur || *cur <= frame, "L1 activation counter moved backwards");
    current_names.set(rec, frame);
  }

  void get_original_name(expr2tc &expr) const override
  {
    renaming_levelt::get_original_name(expr, symbol_renaming_level::level0);
  }

  unsigned int current_number(const irep_idt &name) const;

  level1t() = default;
  ~level1t() override = default;

  virtual void print(std::ostream &out) const;
};

// level 2 -- SSA

struct level2t : public renaming_levelt
{
protected:
  virtual void
  coveredinbees(expr2tc &lhs_sym, unsigned count, unsigned node_id);

public:
  class name_record
  {
  public:
    name_record()
    {
      compute_hash();
    }

    name_record(const symbol2t &sym)
      : base_name(sym.thename),
        lev(sym.rlevel),
        l1_num(sym.level1_num),
        t_num(sym.thread_num)
    {
      compute_hash();
    }

    int compare(const name_record &ref) const
    {
      if (hash < ref.hash)
        return -1;
      if (hash > ref.hash)
        return 1;

      if (base_name < ref.base_name)
        return -1;
      if (ref.base_name < base_name)
        return 1;

      if (lev < ref.lev)
        return -1;
      if (lev > ref.lev)
        return 1;

      if (l1_num < ref.l1_num)
        return -1;
      if (l1_num > ref.l1_num)
        return 1;

      if (t_num < ref.t_num)
        return -1;
      if (t_num > ref.t_num)
        return 1;

      return 0;
    }

    bool operator<(const name_record &ref) const
    {
      if (compare(ref) == -1)
        return true;
      return false;
    }

    bool operator==(const name_record &ref) const
    {
      if (compare(ref) == 0)
        return true;
      return false;
    }

    irep_idt base_name;
    symbol2t::renaming_level lev = symbol2t::renaming_level::level0;
    unsigned int l1_num = 0;
    unsigned int t_num = 0;

    // Derived from the fields above; used as the fast-path primary key in
    // compare() and by name_rec_hash. compare() short-circuits on it, so it
    // must stay a pure function of them — compute_hash() is the only writer.
    size_t hash = 0;

  private:
    void compute_hash()
    {
      size_t seed = 0;
      esbmct::hash_combine(seed, base_name.get_no());
      esbmct::hash_combine(seed, (uint8_t)lev);
      esbmct::hash_combine(seed, l1_num);
      esbmct::hash_combine(seed, t_num);
      hash = seed;
    }
  };

  struct name_rec_hash
  {
    size_t operator()(const name_record &ref) const
    {
      return ref.hash;
    }

    bool operator()(const name_record &ref, const name_record &ref2) const
    {
      return ref < ref2;
    }
  };

public:
  virtual void make_assignment(
    expr2tc &lhs_symbol,
    const expr2tc &constant_value,
    const expr2tc &assigned_value);

  void rename(expr2tc &expr) override;
  virtual void rename(expr2tc &expr, unsigned count) = 0;

  void get_ident_name(expr2tc &symbol) const override;

  void remove(const expr2tc &symbol) override
  {
    current_names.erase(name_record(to_symbol2t(symbol)));
  }

  /// Retire a name whose storage has gone out of scope. L1 names are never
  /// reused (symex_decl draws from a monotone per-identifier counter), so a
  /// popped frame's local can still be named -- through a dangling pointer --
  /// after teardown. Erasing the record restarts the counter, letting such a
  /// write re-issue an index the declaration already defined and so define one
  /// SSA name twice (I10). Advancing past the last live index instead leaves
  /// the current name undefined, which keeps a read of the expired storage
  /// unconstrained exactly as erasure did.
  void retire(const name_record &rec)
  {
    valuet &entry = current_names[rec];
    ++entry.count;
    entry.constant = expr2tc();
  }

  /// Record `rec` at its initial version. phi_function merges only names that
  /// already had a record when the branch was taken, so storage first written
  /// inside a branch would otherwise keep that branch's version on both paths
  /// (#6798). get_ident_name numbers a count-0 record exactly as it numbers an
  /// absent one, so declaring costs no SSA renumbering.
  void declare(const name_record &rec)
  {
    current_names.emplace(rec, valuet());
  }

  void get_original_name(expr2tc &expr) const override
  {
    renaming_levelt::get_original_name(expr, symbol_renaming_level::level1);
  }

  struct valuet
  {
    unsigned count;
    expr2tc constant;
    unsigned node_id;
    valuet() : count(0), node_id(0)
    {
    }
  };

  unsigned current_number(const expr2tc &sym) const;
  unsigned current_number(const name_record &rec) const;

  // static method to rename a (l0) variable to the l1 number record specified
  // in the given name_record. The use case for this is phi_function, where
  // we have a handle on name_record's identifying the storage variable that
  // we want to assign to, but lack the ability to address it as a symbol.
  // In that case (or any similar) we need a facility independent of a
  // specific level2t object.
  static void rename_to_record(expr2tc &sym, const name_record &rec);

  level2t() = default;
  ~level2t() override = default;
  virtual std::shared_ptr<level2t> clone() const = 0;

  virtual void print(std::ostream &out) const;
  virtual void dump() const;

  friend void build_goto_symex_classes();
  // Repeat of the above ignored friend directive.
  typedef std::unordered_map<name_record, valuet, name_rec_hash> current_namest;

  current_namest current_names;
};

} // namespace renaming

#endif /* _GOTO_SYMEX_RENAMING_H_ */
