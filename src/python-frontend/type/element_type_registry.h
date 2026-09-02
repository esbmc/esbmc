#pragma once

#include <util/irep/type.h>

#include <array>
#include <cassert>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class type_handler;

/// Selects which per-instance element-type sequence a registry query addresses.
///
/// Each slot is a separate namespace: a lookup in one slot can never observe
/// entries recorded in another. The dict slots rely on that isolation — see
/// python_dict_handler for why recording value types where the values-list's
/// own element types live would misread dict storage (github_3719_4).
enum class type_slot
{
  /// The container's own elements: lists, sets, tuples, and a dict's internal
  /// keys/values lists.
  elements,
  /// A literal dict's per-value types, keyed by its values-list symbol id.
  /// Lets a subscript read detect a heterogeneous int/float dict.
  dict_value_types,
  /// The uniform element type of a literal dict's list-typed values, keyed by
  /// its values-list symbol id, so a d[k] list read can type inner elements.
  dict_value_list_elems,
  /// Number of slots. Not a slot itself.
  slot_count,
};

/// The single owner of per-instance element-type tracking for the Python
/// frontend: "which types does this container instance hold?".
///
/// One instance lives on python_converter, so the tracked state has the
/// lifetime of a conversion rather than of the process. Every container
/// operation that must forward element types through it — a slice, a sort, a
/// pop, a dict subscript — consults this registry instead of reaching into a
/// shared map, so a new operation has one place to preserve type information.
///
/// Invariant: an instance is either absent or has at least one recorded entry.
/// Only record()/assign_from()/append_from() create entries and each adds at
/// least one, so queries never observe a present-but-empty instance. find()
/// therefore answers both "is anything recorded?" and "is this known?".
class element_type_registry
{
public:
  /// A recorded element: its symbol id, empty when the type came from an
  /// annotation rather than a concrete element expression, and its type.
  using entry = std::pair<std::string, typet>;
  using entries = std::vector<entry>;

  void record(
    const std::string &id,
    const std::string &elem_id,
    const typet &elem_type,
    type_slot slot = type_slot::elements);

  /// Replaces @p to's entries with @p from's, leaving @p to untouched when
  /// @p from has nothing recorded.
  void assign_from(
    const std::string &from,
    const std::string &to,
    type_slot slot = type_slot::elements);

  /// Appends @p from's entries onto @p to's, @p times over. @p from is read
  /// once, so `to == from` (a `l = l * n` repetition) repeats the original
  /// sequence rather than compounding what earlier repetitions appended.
  void append_from(
    const std::string &from,
    const std::string &to,
    size_t times = 1,
    type_slot slot = type_slot::elements);

  /// Removes the most recently recorded entry, mirroring a pop from the end.
  /// Erases the instance when its last entry goes, preserving the invariant.
  void pop_last(const std::string &id, type_slot slot = type_slot::elements);

  /// Mirrors an in-place reversal so later index-based lookups stay correct.
  /// A no-op for an unknown or single-element instance.
  void reverse(const std::string &id, type_slot slot = type_slot::elements);

  size_t
  size(const std::string &id, type_slot slot = type_slot::elements) const;

  /// The recorded entries, or nullptr when nothing is recorded.
  const entries *
  find(const std::string &id, type_slot slot = type_slot::elements) const;

  /// The type at @p index, falling back to the first element's type when
  /// @p index is out of bounds. Nil when nothing is recorded.
  typet element_type(
    const std::string &id,
    size_t index = 0,
    type_slot slot = type_slot::elements) const;

  /// The last recorded element's type; nil when nothing is recorded.
  typet last_element_type(
    const std::string &id,
    type_slot slot = type_slot::elements) const;

  /// The symbol id at @p index; empty when out of bounds or unrecorded.
  /// Unlike element_type(), an out-of-bounds index does not fall back.
  std::string element_id(
    const std::string &id,
    size_t index,
    type_slot slot = type_slot::elements) const;

  /// True when both integer and floating-point elements are recorded.
  bool has_mixed_numeric(
    const std::string &id,
    type_slot slot = type_slot::elements) const;

  /// The shared type of every recorded element, or nil when nothing is
  /// recorded or the elements disagree.
  typet uniform_element_type(
    const std::string &id,
    type_slot slot = type_slot::elements) const;

  /// Non-throwing element type of an all-numeric instance: double_type() when
  /// int and float mix (Python promotes int to float), the single shared
  /// integer type when every element shares it, otherwise nil — including for
  /// any non-numeric element or integers of differing widths.
  typet numeric_element_type(
    const std::string &id,
    type_slot slot = type_slot::elements) const;

  /// The common element type for an operation requiring one, promoting an
  /// int/float mix to double_type() as Python does. Nil when nothing is
  /// recorded.
  /// @throws std::runtime_error when the elements mix incompatibly, naming
  ///         @p func_name.
  typet homogeneous_element_type(
    const std::string &id,
    const std::string &func_name,
    type_slot slot = type_slot::elements) const;

  /// The element type already answered for one syntactic pop() site, or nil
  /// when that site has not been answered yet. The assignment path converts
  /// its RHS more than once, and a pop consumes a recorded entry, so the
  /// second conversion must replay the first answer rather than consume
  /// another entry (#4780).
  typet memoized_pop_type(const std::string &site) const;

  void memoize_pop_type(const std::string &site, const typet &elem_type);

  /// Computes the type_flag and float_type_id encoding shared with
  /// __ESBMC_list_sort and __ESBMC_list_lt:
  ///   0 = all-integer, 1 = all-float, 2 = string, 3 = mixed int+float.
  /// Inspects one instance only; a caller comparing two operands (e.g. an int
  /// list against a float list) must merge both flags itself.
  void type_flags(
    const std::string &id,
    const type_handler &th,
    int &type_flag,
    size_t &float_type_id,
    type_slot slot = type_slot::elements) const;

private:
  using slot_map = std::unordered_map<std::string, entries>;

  slot_map &map_for(type_slot slot)
  {
    assert(slot < type_slot::slot_count);
    return maps_[static_cast<size_t>(slot)];
  }

  const slot_map &map_for(type_slot slot) const
  {
    assert(slot < type_slot::slot_count);
    return maps_[static_cast<size_t>(slot)];
  }

  std::array<slot_map, static_cast<size_t>(type_slot::slot_count)> maps_;

  /// Keyed by "<list id>:<line>:<column>" — one syntactic pop() site.
  std::unordered_map<std::string, typet> pop_memo_;
};
