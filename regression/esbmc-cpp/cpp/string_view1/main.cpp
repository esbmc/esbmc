#include <iostream>
#include <string_view>

int main()
{
  // Original test
  std::string_view message = "Hello, World!";

  // Accessing individual characters
  std::cout << "First character: " << message[0] << std::endl;
  std::cout << "Third character: " << message[2] << std::endl;

  // Length of the string view
  std::cout << "Length of message: " << message.length() << std::endl;

  // Checking if empty
  std::string_view emptyView;
  std::cout << "Is emptyView empty? " << (emptyView.empty() ? "Yes" : "No")
            << std::endl;

  // Comparing string views
  std::string_view otherMessage = "Hello, World!";

  return 0;
}
