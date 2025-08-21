#include <stdio.h>
#include <stdlib.h>

typedef struct Node{
    int data;
    struct Node* next;

}Node;

Node* createNode(int data){
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

void insertNode(Node** head,int data){
    if(*head == NULL){
        *head = createNode(data);
    }else{
        Node* temp = *head;
        while(temp->next != NULL){
            temp = temp->next;
        }
        temp->next = createNode(data);
    }
}
int existINList(Node* head,int data){
    Node* temp = head;
    while(temp != NULL){
        if(temp->data == data){
            return 1;
        }
        temp = temp->next;
    }
    return 0;
}
Node* andList(Node* A,Node* B){
    Node* C = NULL;
    Node* temp = A;
    while(temp != NULL){
        if(existINList(B,temp->data)){
            insertNode(&C,temp->data);
        }
        temp = temp->next;
    }
    return C;
}
void printList(Node* head){
    Node* temp = head;
    while(temp != NULL){
        printf("%d ",temp->data);
        temp = temp->next;
    }
    printf("\n");
}
int main(){
    Node* A = NULL;
    Node* B = NULL;
    insertNode(&A,1);
    insertNode(&A,2);
    insertNode(&A,3);
    insertNode(&A,4);
    insertNode(&A,5);
    insertNode(&A,6);
    insertNode(&A,7);
    insertNode(&A,8);
    insertNode(&A,9);
    insertNode(&A,10);
    insertNode(&B,1);
    insertNode(&B,3);
    insertNode(&B,5);
    insertNode(&B,7);
    insertNode(&B,9);
    Node* C = andList(A,B);
    printList(C);
    return 0;
}
    
    
