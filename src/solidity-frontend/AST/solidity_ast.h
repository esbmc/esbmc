#include <util/message/default_message.h>

// For json parsing
using json = nlohmann::json;

class solidity_ast
{
public:
  /**
   * @brief Prints the contents into the
   * the stdout.
   */
  void dump() const
  {
    default_message msg;
    msg.debug(this->to_string());
  }
}