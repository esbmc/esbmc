#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>

#include <optional>
#include <string>
#include <vector>

class python_converter;

/// One operand of an nditer: a named array or view and its layout in
/// elements (a negative stride walks backwards, 0 repeats a broadcast axis).
/// A 1-D view whose extent or stride is only known at run time keeps them as
/// expressions in `extent` and `stride`.
struct numpy_nditer_operand
{
  nlohmann::json array;
  std::vector<long long> shape;
  std::vector<long long> strides;
  nlohmann::json extent;
  nlohmann::json stride;
};

/// The static description of an `np.nditer`: its operands, the iteration
/// shape, and the order the positions are visited in. The position itself
/// lives in hidden Python variables named after `state`.
struct numpy_nditer_state
{
  std::string state;
  /// The name numpy is imported under, for the arrays an external loop
  /// yields.
  std::string alias;
  std::vector<numpy_nditer_operand> operands;
  std::vector<long long> shape;
  /// Iteration axes, outermost first.
  std::vector<std::size_t> axes;
  /// Axes visited from their last index down to 0.
  std::vector<bool> flipped;
  /// A 1-D iteration whose extent and direction are run-time values.
  bool runtime = false;
  bool flip_negative = false;
  bool multi_index = false;
  bool external_loop = false;
  bool buffered = false;
  /// Elements per item: the chunk an external loop yields, 1 otherwise.
  long long chunk = 1;
  long long size = 0;
};

/// A name bound to what an nditer yields: the element (or the external-loop
/// chunk) of one operand at the position captured when it was bound. Reads
/// go to the operand, so the name is a view as in NumPy.
struct numpy_nditer_item
{
  std::string iterator;
  std::size_t operand = 0;
  std::string position;
};

/// `np.nditer` as an iterator object over named arrays and views
/// (ADR-NP-003): the converter keeps its position in hidden variables and
/// lowers every read through the operand's own subscript, in the order NumPy
/// visits it (C, F, or K: memory order, negative strides undone).
class numpy_nditer
{
public:
  explicit numpy_nditer(python_converter &converter);

  static bool
  is_nditer_call(const nlohmann::json &node, const nlohmann::json &ast);

  /// `name = np.nditer(...)`: validates the call and creates the iterator.
  void create(const std::string &name, const nlohmann::json &call);

  /// The elements a bare `np.nditer(...)` yields, in visiting order.
  exprt as_list(const nlohmann::json &call);

  /// `target = <iterator>.<...>` forms that bind what the iterator yields;
  /// false when `value` is not one of them.
  bool bind(const nlohmann::json &target, const nlohmann::json &value);

  std::optional<exprt> attribute(const nlohmann::json &node);
  std::optional<exprt> call(const nlohmann::json &node);
  std::optional<exprt> read_item(const nlohmann::json &name);
  std::optional<exprt> subscript(const nlohmann::json &node);

  static std::string key(const std::string &function, const std::string &name);

private:
  python_converter &converter_;

  const numpy_nditer_state *find(const nlohmann::json &node) const;
  const numpy_nditer_item *find_item(const nlohmann::json &node) const;

  std::optional<numpy_nditer_state>
  describe(const nlohmann::json &call, const std::string &state);
  numpy_nditer_operand operand(const nlohmann::json &node, std::size_t index);
  nlohmann::json named_operand(const nlohmann::json &node, std::size_t index);
  /// `Info` is python_converter's view description, complete only where
  /// these are defined.
  template <class Info>
  void view_layout(const Info &info, numpy_nditer_operand &result);
  template <class Info>
  void runtime_layout(const Info &info, numpy_nditer_operand &result);
  std::string invalid_setup(
    const std::vector<std::string> &flag_list,
    const std::string &order_name) const;
  bool writable_operands_fit(
    const numpy_nditer_state &it,
    const std::vector<std::vector<std::string>> &op_flags);
  bool check_size(numpy_nditer_state &it, bool zerosize_ok);
  std::vector<std::string> flags(const nlohmann::json &call);
  std::string order(const nlohmann::json &call);
  void order_axes(numpy_nditer_state &it, const std::string &order) const;
  void set_chunk(numpy_nditer_state &it) const;
  bool broadcast(numpy_nditer_state &it);

  nlohmann::json position(const numpy_nditer_state &it) const;
  nlohmann::json items(const numpy_nditer_state &it) const;
  nlohmann::json finished(const numpy_nditer_state &it) const;
  static std::vector<long long> reported_shape(const numpy_nditer_state &it);
  std::vector<nlohmann::json> indices(
    const numpy_nditer_state &it,
    const nlohmann::json &element_position) const;
  nlohmann::json element(
    const numpy_nditer_state &it,
    std::size_t op,
    const nlohmann::json &element_position) const;
  exprt
  chunk_value(const numpy_nditer_state &it, const numpy_nditer_item &item);
  nlohmann::json prebuilt(const exprt &expr);
  nlohmann::json materialize(const numpy_nditer_item &item);
  bool substitute_chunks(nlohmann::json &node);

  void assign(const std::string &name, const nlohmann::json &value);
  void raise(const std::string &exception, const std::string &message);
  void raise_if(
    const nlohmann::json &cond,
    const std::string &exception,
    const std::string &message);
  void advance(const numpy_nditer_state &it);
  void bind_name(
    const nlohmann::json &target,
    const numpy_nditer_state &it,
    const std::string &iterator,
    std::size_t op);
  void bind_operand(
    const nlohmann::json &target,
    const numpy_nditer_state &it,
    const std::string &iterator,
    const nlohmann::json &index);
  void bind_items(
    const nlohmann::json &target,
    const numpy_nditer_state &it,
    const std::string &iterator);
  nlohmann::json current_values(const numpy_nditer_state &it) const;
  nlohmann::json
  layout_attribute(const numpy_nditer_state &it, const std::string &attr) const;
  nlohmann::json
  position_attribute(const numpy_nditer_state &it, const std::string &attr);
  void check_unpack(const numpy_nditer_state &it, long long expected);
  nlohmann::json call_method(
    const numpy_nditer_state &it,
    const std::string &method,
    const nlohmann::json &node);
};
