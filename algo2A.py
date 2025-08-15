class Pila:
    """Clase que implementa una pila (stack) usando una lista"""
    def __init__(self):
        """Inicializa la pila como una lista vacía"""
        self.items = []
    def push(self, item):
        """
        Agrega un elemento a la parte superior de la pila
        Parámetros:
        item (cualquier tipo): Elemento a agregar a la pila
        """
        self.items.append(item)
        print(f"Se agregó '{item}' al historial.")

    def pop(self):
        """
        Elimina y muestra el elemento en la parte superior de la pila
        Si la pila está vacía, imprime un mensaje informandolo
        """
        if not self.is_empty():  # Verifica si la pila no está vacía
            removed_item = self.items.pop()  # Quita el último elemento
            print(f'Elemento "{removed_item}" removido del historial.')
        else:
            print('El historial está vacío.')

    def peek(self):
        """
        Muestra y devuelve el elemento en la cima de la pila sin eliminarlo.
        Si la pila está vacía, muestra un mensaje indicando eso.
        """
        if not self.is_empty():
            top_item = self.items[-1]  # Accede al último elemento
            print(f'El primer item del historial es "{top_item}".')
            return top_item
        else:
            print('El historial está vacío.')

    def is_empty(self):
        """
        Verifica si la pila está vacía
        Retorna:
        bool: True si la pila está vacía, False en caso contrario
        """
        return len(self.items) == 0  # Devuelve el valor booleano

    def show(self):
        """
        Muestra todos los elementos de la pila, desde el fondo hasta la cima.
        Si la pila está vacía, lo indica con un mensaje.
        """
        if self.is_empty():
            print('La pila está vacía.')
        else:
            print('Estado actual del historial (de abajo hacia arriba):')
            for item in self.items:
                print(f'{item}')


# Ejemplo
if __name__ == "__main__":
    historial_de_navegacion = Pila()

    # Agregando sitios al historial
    historial_de_navegacion.push('https://www.microsoft.com/')
    historial_de_navegacion.push('https://web.whatsapp.com/')
    historial_de_navegacion.push('https://open.spotify.com')

    # Eliminando el último sitio visitado
    historial_de_navegacion.pop()
    # Mostrando el último sitio actual
    historial_de_navegacion.peek()
    # Eliminando otro sitio
    historial_de_navegacion.pop()
    # Mostrando el último sitio actual
    historial_de_navegacion.peek()
    # Mostrando el historial actual
    historial_de_navegacion.show()
    # Agregando más sitios
    historial_de_navegacion.push('https://co.pinterest.com/')
    historial_de_navegacion.push('https://github.com/')
    historial_de_navegacion.push('https://www.ilovepdf.com')
    # Eliminando el último sitio actual
    historial_de_navegacion.pop()
    # Mostrando el último sitio actual
    historial_de_navegacion.peek()
    # Mostrando el historial actual
    historial_de_navegacion.show()

    # Vaciando completamente la pila
    historial_de_navegacion.pop()
    historial_de_navegacion.pop()
    historial_de_navegacion.pop()

    # Comprobando el estado final de la pila
    historial_de_navegacion.peek()
    historial_de_navegacion.show()

    # Verificando si la pila está vacía
    print("¿La pila está vacía?", "Sí" if historial_de_navegacion.is_empty() else "No")
