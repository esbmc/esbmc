#pragma once

#include <util/irep.h>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

class namespacet;

/// Closed-world numeric type-id registry for symbolic exception dispatch
/// (issue #5075).
///
/// ESBMC sees the whole program, so the set of exception types and the
/// subtype ("is-a") relation between them are fixed once the GOTO program is
/// built. This table assigns every class type a stable integer id and records
/// each type's direct bases from the symbol table's "bases" metadata, deriving
/// the reflexive-transitive subtype relation on demand. Both the C++ and
/// Python frontends record bases
/// the same way: a "bases" sub on the struct/class type whose entries have
/// id() == "tag-<ClassName>" (see clang_cpp_convertert::get_base_components_
/// methods and goto_symext::is_python_exception_subtype). A single table
/// therefore serves both frontends, giving the issue's "shared semantics"
/// for free.
///
/// The match guard for `catch (C)` becomes the finite disjunction
///   __ESBMC_exc_typeid in concrete_subtype_ids(C)
/// over a *symbolic* thrown type-id; `catch (...)` is just `thrown == true`.
/// The relation is static (closed-world) while the thrown type stays
/// symbolic, which is exactly what makes GOTO-level dispatch fully symbolic.
class exception_typeidt
{
public:
  /// Builds the table by enumerating every type symbol reachable through
  /// @p ns and reading its direct bases. @p ns must outlive this object.
  explicit exception_typeidt(const namespacet &ns);

  /// Reserved id meaning "no in-flight exception type".
  static constexpr unsigned no_type = 0;

  /// Ingest an exception type's ancestry from a THROW's exception_list
  /// (front = dynamic type, rest = its bases, most-derived first). This is how
  /// the Python frontend conveys its exception hierarchy — those types have no
  /// `tag-` symbol, so they are unknown to the symbol-table-built table until
  /// registered here. Idempotent; unions into any existing bases.
  void register_chain(const std::vector<irep_idt> &chain);

  /// Stable id for @p name. Names use the unprefixed convention of
  /// code_cpp_throw2t::exception_list (e.g. "std::bad_cast", never
  /// "tag-std::bad_cast"). A name absent from the program is assigned a fresh
  /// id on first request, so opaque/library types remain dispatchable.
  unsigned id_of(const irep_idt &name);

  /// True iff @p name is an actual program type recorded at construction
  /// (not a primitive `#cpp_type`, opaque, or lazily-minted name). The
  /// lowering pass only matches over registered types, so this gates whether a
  /// throw/catch can be lowered at all.
  bool is_registered(const irep_idt &name) const
  {
    // `void_ptr` is the universal pointer catch — always matchable (it catches
    // any thrown pointer), even when no throw names it explicitly.
    return name == "void_ptr" || name_to_id.find(name) != name_to_id.end();
  }

  /// Reflexive-transitive subtype test: true iff @p thrown equals @p caught
  /// or derives from it transitively through the bases graph. Reproduces the
  /// semantics of goto_symext::is_python_exception_subtype.
  bool is_subtype(const irep_idt &thrown, const irep_idt &caught) const;

  /// { id_of(T) : T <: caught } over all *registered* program types — the
  /// right-hand side of the catch-match guard.
  std::set<unsigned> concrete_subtype_ids(const irep_idt &caught) const;

  /// Number of registered program types (excludes lazily-added unknowns).
  std::size_t size() const
  {
    return registered_count;
  }

private:
  // Deterministic, build-stable name<->id mapping: ids are assigned in sorted
  // name order so they do not depend on symbol insertion order.
  std::map<irep_idt, unsigned> name_to_id;

  // Direct base names (unprefixed) per registered class, from "bases".
  std::map<irep_idt, std::vector<irep_idt>> direct_bases;

  std::size_t registered_count = 0;
  unsigned next_id = no_type + 1;
};
