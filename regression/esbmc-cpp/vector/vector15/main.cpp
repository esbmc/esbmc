#include <vector>
#include <iostream>
using namespace std;
 
void display_sizes(const std::vector<int> &nums1,
                   const std::vector<int> &nums2,
                   const std::vector<int> &nums3)
{
    std::cout << "nums1: " << nums1.size() 
              << " nums2: " << nums2.size()
              << " nums3: " << nums3.size() << '\n';
}
 
int main()
{
    std::vector<int> nums1;
	 nums1.push_back(1);
	 nums1.push_back(2);
	 nums1.push_back(3);
	 nums1.push_back(4);
	 nums1.push_back(5);
    std::vector<int> nums2; 
    std::vector<int> nums3;
 
    std::cout << "Initially:\n";
    display_sizes(nums1, nums2, nums3);
 
    // copy assignment copies data from nums1 to nums2
    nums2 = nums1;
 
    std::cout << "After assignment:\n"; 
    display_sizes(nums1, nums2, nums3);
 
    // move assignment moves data from nums1 to nums3,
    // modifying both nums1 and nums3
    //nums3 = move(nums1);
 
    std::cout << "After move assignment:\n"; 
    display_sizes(nums1, nums2, nums3);
}
