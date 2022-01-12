fun sort(arr: IntArray, N: Int)
{
    for(i in 0..(N-1))
        arr[i] = i+1;
}

fun main() {
    val arr = IntArray(20);
    for(i in 0..19)
      arr[i] = 20 - i;

    sort(arr, 20);

    for(i in 0..19)
      assert(arr[i] == i+1);
}