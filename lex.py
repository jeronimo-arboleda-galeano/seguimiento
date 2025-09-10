import re

# ============================================================
# Definición de expresiones regulares para cada tipo de token
# ============================================================
TOKEN_TYPES = {
    # Palabras reservadas del lenguaje
    'PALABRA_CLAVE': r'\b(if|else|while|return|function)\b',

    # Identificadores (variables, nombres de funciones, etc.)
    'IDENTIFICADOR': r'\b[a-zA-Z_]\w*\b',

    # Números enteros o decimales
    'NUMERO': r'\b\d+(\.\d+)?\b',

    # Operadores (básicos y extendidos: ++, --, &&, ||, +=, etc.)
    'OPERADOR': r'(==|!=|<=|>=|\+\+|--|\+=|-=|\*=|/=|&&|\|\||[+\-*/=<>!&|])',

    # Símbolos de agrupación y separación
    'SIMBOLO': r'[(),{};]',

    # Cadenas de texto con soporte de caracteres de escape
    'CADENA': r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\'',

    # Espacios en blanco (se ignoran en el análisis)
    'ESPACIO_BLANCO': r'\s+',

    # Comentarios de una línea y multilínea
    'COMENTARIO_LINEA': r'//.*',
    'COMENTARIO_MULTI': r'/\*[\s\S]*?\*/'
}

# Clase Token: representa cada unidad léxica encontrada
class Token:
    def __init__(self, type_, value, line, column):
        self.type = type_    # Tipo de token (ej: IDENTIFICADOR, NUMERO, etc.)
        self.value = value   # Texto exacto del token
        self.line = line     # Línea donde aparece
        self.column = column # Columna inicial del token

    def __repr__(self):
        return f"Token(tipo={self.type}, valor={self.value}, linea={self.line}, columna={self.column})"


# Función auxiliar: vista previa de un fragmento del código
def _esc_preview(s, n=24):
    """
    Devuelve los primeros caracteres de un string
    reemplazando saltos de línea y tabulaciones,
    para mostrar en mensajes de depuración o error.
    """
    p = s[:n].replace('\n', '\\n').replace('\t', '\\t')
    if len(s) > n:
        p += '…'
    return p


# Función principal: lexer
def lexer(codigo_fuente, debug=False):
    """
    Analizador léxico.
    Recorre el código fuente y lo transforma en una lista de tokens.
    
    Parámetros:
    - codigo_fuente: string con el código a analizar.
    - debug: si es True, muestra paso a paso el proceso de análisis.
    """

    # Función interna para imprimir mensajes de depuración
    def dbg(msg):
        if debug:
            print(msg)

    tokens = []     # Lista final de tokens
    linea = 1       # Contador de líneas
    columna = 1     # Contador de columnas
    posicion = 0    # Posición actual en el código
    paso = 0        # Contador de pasos (para depuración)

    # Bucle principal: recorre todo el código fuente
    while posicion < len(codigo_fuente):
        paso += 1
        dbg(f"[{paso:04}] pos={posicion}, linea={linea}, columna={columna}, mirar='{_esc_preview(codigo_fuente[posicion:])}'")

        match_text = None
        matched_type = None

        # Intentar matchear con cada expresión regular de TOKEN_TYPES
        for token_type, regex in TOKEN_TYPES.items():
            regex_match = re.match(regex, codigo_fuente[posicion:])
            if regex_match:
                match_text = regex_match.group(0)
                matched_type = token_type
                break  # primera coincidencia encontrada

        # Si hubo coincidencia, procesar el token
        if match_text is not None:
            if matched_type in ('ESPACIO_BLANCO', 'COMENTARIO_LINEA', 'COMENTARIO_MULTI'):
                # Omitimos espacios y comentarios (no generan tokens)
                nls = match_text.count('\n')
                if matched_type == 'ESPACIO_BLANCO':
                    dbg(f"   -> Omitir {matched_type} len={len(match_text)} (nl={nls})")
                else:
                    dbg(f"   -> Omitir {matched_type}")

                # Actualizar posición y contadores de línea/columna
                if nls:
                    linea += nls
                    last_nl = match_text.rfind('\n')
                    columna = 1 + (len(match_text) - last_nl - 1)
                else:
                    columna += len(match_text)
                posicion += len(match_text)

            else:
                # Generamos un token válido
                dbg(f"   -> Token {matched_type} = '{_esc_preview(match_text, 40)}'")
                tokens.append(Token(matched_type, match_text, linea, columna))
                posicion += len(match_text)
                columna += len(match_text)

            continue  # Volver al inicio del while para procesar lo siguiente

        # Si no hubo coincidencia, es un error léxico
        ch = codigo_fuente[posicion]
        contexto = _esc_preview(codigo_fuente[posicion:posicion+40], 40)

        if ch in ('"', "'"):
            dbg("   !! Error: cadena sin cerrar")
            raise SyntaxError(f"Error léxico: cadena sin cerrar en línea {linea}, columna {columna}. Contexto: '{contexto}'")

        if codigo_fuente[posicion:posicion+2] == '/*':
            dbg("   !! Error: comentario multilínea sin cerrar")
            raise SyntaxError(f"Error léxico: comentario multilínea sin cerrar en línea {linea}, columna {columna}. Contexto: '{contexto}'")

        if not ch.isprintable():
            dbg("   !! Error: carácter no imprimible")
            raise SyntaxError(f"Error léxico: carácter no imprimible en línea {linea}, columna {columna}. (ord={ord(ch)})")

        dbg(f"   !! Error: carácter inesperado '{ch}'")
        raise SyntaxError(f"Error léxico: carácter inesperado '{ch}' en línea {linea}, columna {columna}. Contexto: '{contexto}'")

    dbg(f"[FIN] Tokens generados: {len(tokens)}")
    return tokens

# Ejemplo de uso

if __name__ == "__main__":
    source_code = ''' 
function add(x, y) { 
    return x + y; 
}

/* Comentario
   multilínea */
if (x >= 10 && y != 5) { 
    y += 2;
    mensaje = "Hola\\nMundo";
    // línea comentada
} 
'''

    # Modo normal
    tokens = lexer(source_code, debug=False)
    print("=== TOKENS ENCONTRADOS ===")
    for t in tokens:
        print(t)

    print("\n=== MODO DEBUG ===")
    # Modo depuración (muestra el proceso paso a paso)
    _ = lexer(source_code, debug=True)
