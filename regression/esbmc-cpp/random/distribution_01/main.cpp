#include <vector>
#include <random>
#include <cassert>
#include <iostream>

int counter = 0;

int sum_with_side_effect(const std::vector<int> &numbers)
{
  int total = 0;
  for (int num : numbers)
  {
    total += num;
  }
  counter++; // side effect
  return total;
}

int sum_no_side_effect(const std::vector<int> &numbers)
{
  int total = 0;
  for (int num : numbers)
  {
    total += num;
  }
  return total;
}

int main()
{
  // Create random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> value_dist(1, 100);
  std::uniform_int_distribution<int> size_dist(1, 10);

  // Create random vector (equivalent to Python list comprehension)
  int size = size_dist(gen);
  std::vector<int> x;
  for (int i = 0; i < size; ++i)
  {
    x.push_back(value_dist(gen));
  }

  // Assert that both functions return the same result
  assert(sum_with_side_effect(x) == sum_no_side_effect(x));

  std::cout << "Assertion passed!" << std::endl;

  return 0;
}
