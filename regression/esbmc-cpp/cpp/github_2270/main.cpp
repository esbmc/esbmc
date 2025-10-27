#include <iostream>
#include <vector>
#include <string>
#include <cassert>

class Task {
public:
    std::string name;
    std::vector<Task> subtasks;

    Task(std::string name) : name(name) {}

    void add_subtask(const Task& task) {
        subtasks.push_back(task);
    }

    std::string to_string() const {
        std::string result = "Task(" + name + ")";
        if (!subtasks.empty()) {
            result += " with subtasks: [";
            for (size_t i = 0; i < subtasks.size(); ++i) {
                result += subtasks[i].to_string();
                if (i < subtasks.size() - 1) {
                    result += ", ";
                }
            }
            result += "]";
        }
        return result;
    }
};

void demonstrate_recursive_tasks() {
    // Create main task
    Task main_task("Main");

    // Add some subtasks
    Task sub1("Subtask 1");
    Task sub2("Subtask 2");
    main_task.add_subtask(sub1);
    main_task.add_subtask(sub2);

    // Add a sub-subtask
    Task sub_sub("Sub-subtask");
    sub1.add_subtask(sub_sub);

    // Assertions
    assert(main_task.name == "Main");
    assert(main_task.subtasks.size() == 2);
    assert(main_task.subtasks[0].name == "Subtask 1");
    assert(main_task.subtasks[1].name == "Subtask 2");
    assert(sub1.subtasks.size() == 1);
    assert(sub1.subtasks[0].name == "Sub-subtask");

    std::cout << main_task.to_string() << std::endl;
}

int main() {
    demonstrate_recursive_tasks();
    return 0;
}

