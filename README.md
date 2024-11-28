# AA1-TUIA-Zahradnicek
Repositorio creado para la entrega del TP2 de la materia de Aprendizaje Automático 1 - TUIA

Este repositorio contiene el código y las configuraciones necesarias para crear una imagen Docker que ejecuta un modelo de inferencia del Trabajo Práctico 2 de MLOps.

## Requisitos previos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

- **Docker**: [Descargar e instalar Docker](https://www.docker.com/get-started)

## Construcción de la imagen Docker

Para construir la imagen Docker, abre una terminal en el directorio donde se encuentra el archivo `Dockerfile` y ejecuta el siguiente comando:

```bash
docker build -t inference-tp2-mlops .
```

## Ejecutar el contenedor

Para ejecutar el contenedor, utiliza el siguiente comando. Asegúrate de reemplazar "ruta-de-tu-carpeta" con la ruta de la carpeta en tu máquina local que tiene el archivo 'input.csv'.

```bash
docker run -it --rm --name inference-tp2-mlops-container -v "ruta-de-tu-carpeta:/files" inference-tp2-mlops
```