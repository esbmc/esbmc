package root.foo
import root.sort.sort
import root.sort.is_sorted

fun initialize_array(arr: IntArray, N: Int)
{
    for(i in 0..(N-1))
      arr[i] = N - i;
}

fun main() {
    val arr = IntArray(20);
    initialize_array(arr, 20);

    sort(arr, 20);

    assert(is_sorted(arr,20));
}