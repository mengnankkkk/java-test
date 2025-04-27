#include <stdio.h>

int partition(int arr[],int n,int pivoIndex){
    int pivot = arr[pivoIndex];
    int left = 0;
    int right = n-1;


    int temp = arr[pivoIndex];
    arr[pivoIndex] = arr[right];
    arr[right] = temp;

    while(left<right){
        while(left<right && arr[left]<pivot){
            left++;
        }
        while(left<right && arr[right]>=pivot){
            right--;
        }
        if(left<right){
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
        }
    }
    arr[n-1] = arr[left];
    arr[left] = pivot;
    return left;
}