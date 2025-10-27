#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cassert>

int main()
{
    // Example 1: Track unique student IDs using unordered_set
    std::unordered_set<int> student_ids;
    
    // Add some student IDs
    student_ids.insert(101);
    student_ids.insert(102);
    student_ids.insert(103);
    student_ids.insert(101); // Duplicate - won't be added
    // Assertion: Should have exactly 3 unique elements
    assert(student_ids.size() == 3);
    assert(!student_ids.empty());
    
    // Check if a student ID exists
    auto found_it = student_ids.find(102);
    assert(found_it != student_ids.end()); // Should find student 102
    assert(student_ids.count(102) == 1);   // Alternative check
    assert(student_ids.contains(102));     // C++20 style check    
    // Check for non-existent student
    auto not_found_it = student_ids.find(999);
    assert(not_found_it == student_ids.end()); // Should not find student 999
    assert(student_ids.count(999) == 0);
    assert(!student_ids.contains(999));
    
    // Example 2: Map student IDs to their grades using unordered_map
    std::unordered_map<int, int> student_grades;
    
    // Assign grades
    student_grades[101] = 10;
    student_grades[102] = 9;
    student_grades[103] = 8;
    student_grades[104] = 7;
    
    // Assertions: Should have exactly 4 graded students
    assert(student_grades.size() == 4);
    assert(!student_grades.empty());
    
    // Verify specific grades
    assert(student_grades[101] == 10);
    assert(student_grades[102] == 9);
    assert(student_grades[103] == 8);
    assert(student_grades[104] == 7);
    
    // Look up a specific grade
    auto grade_it = student_grades.find(102);
    assert(grade_it != student_grades.end()); // Should find student 102
    assert(grade_it->second == 9);          // Should have grade 'B'
    assert(student_grades.count(102) == 1);   // Alternative check
    assert(student_grades.contains(102));     // C++20 style check
    
    // Check for non-existent student in grades
    auto no_grade_it = student_grades.find(999);
    assert(no_grade_it == student_grades.end()); // Should not find student 999
    assert(student_grades.count(999) == 0);
    assert(!student_grades.contains(999));
    
    return 0;
}
