#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

class Task {
public:
    std::string name;
    int priority;

    Task(std::string name, int priority = 0) : name(name), priority(priority) {}

    std::string to_string() const {
        return "Task(" + name + ", priority=" + std::to_string(priority) + ")";
    }
};

void demonstrate_lists() {
    // Create list of tasks
    std::vector<Task> tasks;

    // Add some tasks
    tasks.push_back(Task("First Task", 1));
    tasks.push_back(Task("Second Task", 2));
    tasks.push_back(Task("Priority Task", 3));

    // Assertions
    assert(tasks.size() == 3);
    assert(tasks[0].name == "First Task" && tasks[0].priority == 1);
    assert(tasks[1].name == "Second Task" && tasks[1].priority == 2);
    assert(tasks[2].name == "Priority Task" && tasks[2].priority == 3);

    // Demonstrate sorting with custom key
    std::vector<Task> sorted_tasks = tasks;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), [](const Task &a, const Task &b) {
        return a.priority > b.priority;
    });

    // Assertions for sorting
    assert(sorted_tasks[0].name == "Priority Task");
    assert(sorted_tasks[1].name == "Second Task");
    assert(sorted_tasks[2].name == "First Task");

    // Demonstrate removal
    Task removed_task = tasks[1];
    tasks.erase(tasks.begin() + 1);

    // Assertions for removal
    assert(tasks.size() == 2);
    assert(tasks[0].name == "First Task");
    assert(tasks[1].name == "Priority Task");

    std::cout << "All assertions passed successfully!" << std::endl;
}

int main() {
    demonstrate_lists();
    return 0;
}

