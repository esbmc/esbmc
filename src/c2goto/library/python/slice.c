#include "python_types.h"

/**
 * @brief Anchor that pulls `__ESBMC_PySliceObj` into the operational-model
 *        symbol table.
 *
 * The Python frontend materialises slice values directly via `struct_exprt`
 * (see `make_slice_struct_expr` in `python_converter.cpp`) rather than
 * routing through a constructor call, but it still needs the struct's
 * tag to be registered as a type so that `type_handler::get_slice_type()`
 * can find it. Providing one externally-visible function that mentions
 * `PySliceObject` is enough to register the type when this translation
 * unit is linked into the `cprover_library`.
 */
PySliceObject __ESBMC_slice_create(
  long long start,
  long long stop,
  long long step,
  int has_start,
  int has_stop,
  int has_step)
{
  PySliceObject s;
  s.start = start;
  s.stop = stop;
  s.step = step;
  s.has_start = has_start;
  s.has_stop = has_stop;
  s.has_step = has_step;
  return s;
}
