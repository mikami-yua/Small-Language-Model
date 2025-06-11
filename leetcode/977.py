from typing import List
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # reserver
        reserved_head_p=ListNode()
        while head:
            temp_node=ListNode(head.val)
            temp_node.next=reserved_head_p.next
            reserved_head_p.next=temp_node
            head = head.next

        num_set=set()
        ans_p=ListNode()
        reserved_head_p=reserved_head_p.next
        while head and head.val not in num_set:
            head_t_p=head
            reser_t_p=reserved_head_p
            ans_p.next=head_t_p
            if reserved_head_p.val not in num_set :
                head_t_p.next=reser_t_p
                num_set.add(head.val)
                num_set.add(reserved_head_p.val)
                head=head.next
                reserved_head_p=reserved_head_p.next
            else:
                break
        return ans_p.next


