#ifndef ESBMC_CPP_BASE_OFFSET_H
#define ESBMC_CPP_BASE_OFFSET_H

#include "dstring.h"
#include "type.h"
#include "expr.h"
#include "c_types.h"
#include "expr_util.h"
#include "namespace.h"

/**
 * @class cpp_base_offset
 * @brief Provides functionality to calculate the offset in bytes to a base class in a C++ object.
 *
 * This class is designed to assist with the calculation of offsets to base classes
 * within C++ objects, particularly in the context of multiple and virtual inheritance,
 * somewhat the Itanium C++ ABI guidelines. It is mainly used to support pointer adjustment
 * in casts.
 */
class cpp_base_offset
{
public:
  /**
   * Calculates the offset to a base class from a given type in bytes.
   *
   * @param base_name The name of the base class to which the offset is calculated.
   * @param type The type from which the offset to the base class is calculated.
   * @param offset_expr An expression representing the calculated offset.
   * @param ns A namespace object to resolve type names and symbols.
   * @return False if the offset could be calculated, true if any error occurred.
   */
  static bool offset_to_base(
    const dstring &base_name,
    const typet &type,
    exprt &offset_expr,
    const namespacet &ns);
};

#endif //ESBMC_CPP_BASE_OFFSET_H