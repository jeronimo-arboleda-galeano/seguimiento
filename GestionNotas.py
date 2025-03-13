# -*- coding: utf-8 -*-
"""
Sistema de Gestión de notas de un estudiante
=======================================
Programa CLI para registro y análisis de notas de un estudiante.
Desarrollado con enfoque educativo para demostrar:
- Modularización de código
- Manejo de estructuras de datos
- Validación de entradas
- Programación estructurada
- Documentación técnica

Autor: jeronimo arboleda galeano
Fecha: [3/12/2025]

Estructura del programa:
1. Definición de tipos de datos
2. Funciones especializadas
3. Menú interactivo
4. Flujo principal de ejecución
"""

import datetime  # Para el manejo de fechas
from typing import List, Dict  # Para las anotaciones de tipos en las listas y diccionarios

# Definición de tipos personalizados
# Alias para mejorar la legibilidad y mantenimiento del código
Nota = Dict[str, str]  # Representa una nota individual con claves de tipo cadena (fecha, asignatura, calificación)
NotaList = List[Nota]  # Lista que contiene varias notas

def mostrar_menu() -> None:
    """
    Muestra el menú de opciones disponible para el usuario.
    """
    print("\n Gestión de Notas ")
    print("1. Agregar nota")
    print("2. Ver todas las notas")
    print("3. Resumen por asignatura")
    print("4. Reporte general")
    print("5. Salir")

def obtener_opcion_valida() -> int:
    """
    Solicita una opción válida al usuario y la retorna como un entero.
    Si la opción ingresada no es válida, se muestra un mensaje de error y se vuelve a pedir.
    """
    while True:
        try:
            opcion = int(input("Seleccione una opción: "))
            if 1 <= opcion <= 5:
                return opcion  # Retorna la opción válida
            print(" Opción fuera de rango (1-5)")
        except ValueError:
            print(" Ingrese un número válido")

def agregar_nota(notas: NotaList) -> None:
    """
    Solicita los datos de una nueva nota al usuario (asignatura, calificación) y la agrega a la lista de notas.
    """
    print("\n Nueva Nota")
    # Solicita los datos de la asignatura y la calificación
    asignatura = input("Asignatura (ej. algebra, programacion, diseño y edicion): ").capitalize()
    calificacion = float(input("Calificación: "))
    fecha = datetime.date.today().strftime("%Y-%m-%d")  # Fecha actual

    # Creación de un diccionario que representa la nueva nota
    nueva_nota = {
        'fecha': fecha,
        'asignatura': asignatura,
        'calificacion': f"{calificacion:.2f}"  # Formato con 2 decimales
    }

    notas.append(nueva_nota)  # Agrega la nueva nota a la lista
    print(f" Nota de {calificacion:.2f} en {asignatura} registrada")

def mostrar_notas(notas: NotaList, titulo: str) -> None:
    """
    Muestra todas las notas registradas en formato de tabla, con su fecha, asignatura y calificación.
    """
    print(f"\n {titulo}")
    if not notas:
        print("No hay registros")  # Si no hay notas, muestra un mensaje
        return
        
    # Encabezados de la tabla
    print(f"{'Fecha':<12} | {'Asignatura':<15} | {'Calificacion':>10}")
    print("-" * 45)  # Línea de separación

    # Itera sobre cada nota en la lista y la imprime en formato tabla
    for nota in notas:
        print(
            f"{nota['fecha']:<12} | {nota['asignatura']:<15} | {nota['calificacion']:>10}")

def resumen_asignaturas(notas: NotaList) -> None:
    """
    Muestra un resumen del promedio de las calificaciones por asignatura.
    """
    asignaturas = {}  # Diccionario para almacenar el total de calificaciones y el conteo por asignatura
    for nota in notas:
        asignatura = nota['asignatura']
        calificacion = float(nota['calificacion'])

        # Acumulación de calificaciones y conteo de notas por asignatura
        if asignatura in asignaturas:
            asignaturas[asignatura]['total'] += calificacion
            asignaturas[asignatura]['count'] += 1
        else:
            asignaturas[asignatura] = {'total': calificacion, 'count': 1}

    # Almacenamos el resumen de promedios en una lista
    resumen = []
    for asignatura, datos in asignaturas.items():
        # Calculamos el promedio de las calificaciones por asignatura
        promedio = datos['total'] / datos['count']
        # Agregamos el resultado formateado a la lista de resumen
        resumen.append(f"- {asignatura}: {promedio:.2f} (Promedio de {datos['count']} notas)")

    # Imprimimos todo el resumen al final
    print("\nResumen por Asignatura")
    if not asignaturas:
        print("No hay datos para mostrar")  # Si no hay asignaturas, muestra un mensaje
    else:
        print("\n".join(resumen))  # Imprime todos los resultados en una sola llamada

def reporte_general(notas: NotaList) -> None:
    """
    Muestra un reporte general con la calificación más baja, el promedio de todas las calificaciones y la calificación más alta.
    """
    if not notas:
        print("\nNo hay notas registradas")  # Si no hay notas, muestra un mensaje
        return

    # Convertimos las calificaciones a flotantes para realizar los cálculos
    calificaciones = [float(n['calificacion']) for n in notas]

    # Cálculos estadísticos
    total = sum(calificaciones)  # Suma total de todas las calificaciones
    minimo = min(calificaciones)  # Calificación más baja
    promedio = total / len(calificaciones)  # Promedio de todas las calificaciones
    max_calificacion = max(calificaciones)  # Calificación más alta

    # Formato del reporte
    print("\n Reporte general")
    print(f"Calificación más baja: {minimo:.2f}")
    print(f"Promedio de todas las notas: {promedio:.2f}")
    print(f"Calificación más alta: {max_calificacion:.2f}")

def main():
    """
    Función principal que inicia el programa y permite interactuar con el usuario.
    """
    # Datos de ejemplo para demostración
    notas_ejemplo = [
        {'fecha': '2025-12-03', 'asignatura': 'algebra', 'calificacion': '4'},
        {'fecha': '2025-12-03', 'asignatura': 'programacion', 'calificacion': '1.5'},
        {'fecha': '2025-12-03', 'asignatura': 'diseño y edicion', 'calificacion': '3.0'}
    ]

    print("¡Bienvenido a tu Gestor de Notas!")
    mostrar_notas(notas_ejemplo, "Notas Pre-cargadas")

    # Bucle principal de interacción con el usuario
    while True:
        mostrar_menu()
        opcion = obtener_opcion_valida()

        # Enrutamiento de las opciones seleccionadas por el usuario
        if opcion == 1:
            agregar_nota(notas_ejemplo)
        elif opcion == 2:
            mostrar_notas(notas_ejemplo, "Todas las Notas")
        elif opcion == 3:
            resumen_asignaturas(notas_ejemplo)
        elif opcion == 4:
            reporte_general(notas_ejemplo)
        elif opcion == 5:
            print("\n¡Hasta luego!")  # Mensaje de despedida
            break

if __name__ == "__main__":
    main()  # Punto de entrada del programa
