# diha

Diagramas de Interacción para elementos de Hormigón Armado

Paquete para el diseño de secciones de hormigón armado

## Installation

### Install from PyPi:

`pip install diha`

Para usar desde la línea de comandos:

`pip install "diha[cli]"`


## Ejemplo de uso por linea de comando:

`diha --help`
```
Usage: diha [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  plot  diha plot Commandline
```

### Imprimir un diagrama
`diha plot --help`
```
Usage: diha plot [OPTIONS] INPUT_FILE

  diha plot Commandline

  Grafica un diagrama de interacción en 2D para una sección predefinida.

Options:
  --output_file TEXT  Nombre del archivo de salida con el grafico del diagrama
  --theta FLOAT       Rotación del vector de momentos en grados respecto del
                      eje +Z  [default: 0]
  --points INTEGER    Numero de puntos a calcular sobre la curva de
                      interacción  [default: 32]
  --debug
  --help              Show this message and exit.

```

Imprimir un diagrama con los valores por defecto

`diha plot section.json`


