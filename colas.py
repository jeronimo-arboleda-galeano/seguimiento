from collections import deque

class queue:
        def __init__(self):
            self.queue= deque()

        def enqueue (self,item):
              """añade un elemento al final de la cola"""
              self.queue.append(item)
              print(f"enqueued :{item}")

        def dequeue(self):
              """elimina y retorna el primer elemento"""    
              if not self.is_empty():
                    item = self.queue.popleft()
                    print(f"dequeued: {item}")
                    return item
              else:
                print("queue is empty!")
                return None
        
        def is_empty(self):
             """verifica si la cola esta vacia"""
             return len(self.queue) == 0
        
        def peek(self):
             """primer elemento"""
             if not self.is_empty():
                  return self.queue[0]
             else:
                  print("queue is empty")

        def size(self):
             """numero de elementos"""
             return len(self.queue)
        
        def display(self):
             """los elementos"""
             print("queue contents:", list(self.queue))

        def clear(self):
             self.queue.clear()
             print("todo libre")
            

queue=queue()

queue.enqueue("imprimir 1 ")
queue.enqueue("imprimir 2 ")
queue.enqueue("imprimir 3 ")

queue.display()
queue.size()
queue.dequeue()

queue.enqueue("imprimir 5 ")
queue.enqueue("imprimir 4 ")

queue.peek()
queue.display()
queue.clear()
queue.display()
queue.peek()
