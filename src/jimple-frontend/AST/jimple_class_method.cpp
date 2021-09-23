#include <jimple-frontend/AST/jimple_class_member.h>

void jimple_class_method::from_json(const json &j)
{
  // Method Name
  j.at("name").get_to(this->name);

  // Method modifiers
  auto modifiers = j.at("modifiers");
  m = modifiers.get<jimple_modifiers>();

  // Method type
  j.at("type").get_to(t);

  // TODO: List of Parameters
  j.at("parameters").get_to(parameters);
  // Throws?
  try {
    j.at("throws").get_to(this->throws);
  }
  catch(std::exception &e)
  {
    this->throws = "(No throw)";
  }

  // TODO: Empty body
  auto j_body = j.at("body");
  // this is a little hacky...
  auto values = j_body.get<jimple_full_method_body>();
  this->body = std::make_shared<jimple_full_method_body>(values);
}
std::string jimple_class_method::to_string() const
{
  std::ostringstream oss;
  oss << "Class Method"
      << "\n\tName: " << this->name
      << "\n\t" << this->t.to_string()
      << "\n\t" << this->m.to_string()
      << "\n\tParameters: " << this->parameters
      << "\n\tThrows: " << this->throws
      << "\n\tBody : " << this->body->to_string();

  return oss.str();
}
