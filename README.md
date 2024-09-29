# splitter-taxdown
# Índice
1. [Introducción](#1-introducción)
2. [Análisis de Datos](#2-análisis-de-datos)
3. [Diseño del Sistema](#3-diseño-del-sistema)
4. [Implementación del Sistema](#4-implementación-del-sistema)
5. [Pruebas Realizadas](#5-pruebas-realizadas)
6. [Análisis del Resultado](#6-análisis-del-resultado)
7. [Análisis del Código](#7-análisis-del-código)
8. [Vídeo Desarrollo](#8-vídeo-desarrollo)

   
## 1. Introducción

Este repositorio contiene la implementación de un sistema "splitter" diseñado para identificar y organizar las preguntas contenidas en los mensajes de usuarios.

A lo largo de este documento, se detallan los pasos seguidos para desarrollar la solución, comenzando con un análisis exhaustivo de los datos, seguido por el diseño y desarrollo del sistema, la exploración de diferentes enfoques, y la evaluación de los resultados obtenidos. Se presentan pruebas y comparaciones entre varios modelos y configuraciones, además de un análisis del código empleado en la implementación.

Para proporcionar una visión más clara del proceso de desarrollo, se ha grabado un vídeo que documenta cómo se llevó a cabo la implementación. El vídeo, sin audio, se enfoca en mostrar mi proceso de trabajo y documentación, ilustrando cómo implementé el código necesario para la solución previamente seleccionada.

## 2. Análisis de Datos

En esta sección se presentan las conclusiones extraídas tras un análisis exploratorio del dataset proporcionado. Para ello, se hizo uso de la librería Pandas y de una visualización de los datos sobre el propio csv original. Se observaron las siguientes características:

- **Variedad de mensajes:** Los mensajes varían significativamente en contenido, longitud y estructura, lo que añade complejidad a su procesamiento.
- **Preguntas múltiples e implícitas:** Los usuarios a menudo formulan múltiples preguntas en un solo mensaje. A veces, estas preguntas están implícitas y no se formulan directamente.
- **Numeración inconsistente:** Algunos mensajes contienen preguntas numeradas, mientras que otros no siguen ninguna estructura específica.
- **Contexto disperso:** La información contextual puede aparecer en cualquier parte del mensaje, lo que dificulta su identificación y separación del contenido relevante.
- **Variabilidad en la redacción:** El nivel y estilo de escritura varían considerablemente entre los usuarios, con algunos mensajes más formales que otros.
- **Uso de conectores:** Los usuarios tienden a utilizar conectores como "además", "por otra parte" o "también" cuando formulan preguntas adicionales.
- **Saludo y despedida:** La mayoría de los mensajes incluyen elementos de saludo o despedida, lo que puede ser relevante para su procesamiento.

## 3. Diseño del Sistema

A continuación se exploran varias posibles soluciones y se analizan sus ventajas y desventajas para abordar el problema de separar las múltiples preguntas y sus contextos.

### 3.1 Sistema Basado en Reglas de NLP

Este enfoque puede utilizar distintas técnicas de NLP tradicionales, como la tokenización, segmentación de oraciones, etiquetado gramatical, expresiones regulares y heurísticas personalizadas para separar las preguntas en los mensajes.

#### Ventajas:
- **Bajo coste computacional:** Los sistemas basados en reglas suelen ser más ligeros y menos costosos de implementar.
- **Rapidez:** La ejecución es rápida debido a la simplicidad del modelo.
- **Alta interpretabilidad:** Las reglas definidas son explícitas y fáciles de entender.

#### Desventajas:
- **Poca flexibilidad:** La rigidez de las reglas hace que este enfoque no se adapte bien a las variaciones en la forma de escribir de los usuarios.
- **Complejidad creciente:** A medida que se añaden más reglas o excepciones, la complejidad del sistema aumenta, lo que dificulta su mantenimiento y escalabilidad.
- **Resultados limitados:** Este enfoque es propenso a errores y es menos efectivo en situaciones complejas con lenguaje natural más variado.

### 3.2 Fine-tuning de un LLM Open-Source

Esta solución implica realizar un fine-tuning de LLM open source para adaptarlo a esta tarea específica. La parte más desafiante sería construir un conjunto de datos de entrenamiento adecuado.

#### Ventajas:
- **Alta calidad de resultados:** Con suficiente entrenamiento, este enfoque puede lograr lor mejores resultados en la identificación y separación de preguntas.
- **Control sobre los datos:** Los datos permanecen en entornos controlados por la empresa, lo que mejora la seguridad y privacidad.
- **Escalabilidad:** Se puede escalar eficientemente si se cuenta con la infraestructura adecuada.
- **Adaptabilidad:** Si se detecta data drift, el modelo puede ser reentrenado.
- **Optimización:** Existen técnicas para optimizar el modelo durante la inferencia, como la cuantización de pesos.
- **Flexibilidad en la entrada:** Capaz de manejar mensajes con estructura y longitud variables.

#### Desventajas:
- **Desarrollo lento:** El proceso de preparar los datos y entrenar el modelo puede ser largo y costoso.
- **Recursos intensivos:** Requiere una mayor cantidad de trabajo e infraestructura para obtener un resultado óptimo.
- **Interpretabilidad limitada:** Similar a otros modelos de deep learning, la salida del modelo puede ser difícil de interpretar.

### 3.3 Uso de un LLM Preentrenado (API)

En esta opción, se utilizan LLMs ya entrenados, como los modelos de OpenAI (por ejemplo GPT-4) a través de una API, para abordar el problema.

#### Ventajas:
- **Resultados sobresalientes:** Los modelos preentrenados suelen proporcionar una excelente calidad en la identificación y separación de preguntas.
- **Desarrollo rápido:** Se puede implementar rápidamente sin necesidad de un proceso de entrenamiento complejo.
- **Escalabilidad:** Al hacer solicitudes a una API, es sencillo ajustar el sistema para manejar grandes volúmenes de mensajes.
- **Optimización de prompts:** Permite ajustar los prompts y parámetros con facilidad para mejorar los resultados.
- **Mantenimiento sencillo:** Las actualizaciones y mejoras de los modelos se gestionan externamente.
- **Flexibilidad en la entrada:** Capaz de manejar entradas variadas en términos de longitud y complejidad.

#### Desventajas:
- **Dependencia externa:** El uso de un servicio externo implica confiar en la estabilidad y precios de la API.
- **Costo por solicitud:** Aunque los precios de las API han disminuido, existe un coste asociado con cada solicitud.
- **Control de datos limitado:** Algunos datos de la empresa se deben compartir con un servicio externo, lo que podría tener implicaciones de privacidad.
- **Interpretabilidad:** La salida del modelo sigue siendo difícil de interpretar.

## 4. Implementación del Sistema
### 4.1 Descripción general

Para esta prueba técnica, se considera que la mejor solución es la 2.3: utilizar un LLM ya entrenado a través de una API, en este caso, los modelos GPT-4o y GPT-4o-mini de OpenAI. Esta elección se basa en su facilidad de implementación, la calidad esperada de los resultados y los menores requisitos de infraestructura.

El objetivo es lograr un "splitter" que organice el contenido de entrada de forma clara y estructurada. El modelo devolverá la información en formato YAML, siguiendo la siguiente estructura:

```yaml
preguntas:
  - pregunta: "Texto de la pregunta 1"
    contexto: "Contexto relacionado con la pregunta 1 (si lo hay)"
  - pregunta: "Texto de la pregunta 2"
    contexto: "Contexto relacionado con la pregunta 2 (si lo hay)"
  - pregunta: "Texto de la pregunta 3"
    contexto: "Contexto relacionado con la pregunta 3 (si lo hay)"
```
__¿Por qué YAML?__

Se ha optado por utilizar YAML en lugar de otros formatos como JSON debido a que el tokenizador de los modelos GPT (BPE implementado en tiktoken) codifica el formato JSON con más tokens adicionales que YAML. Esto genera más ruido para el modelo y aumenta el coste de la solicitud. Al usar YAML, se minimiza el número de tokens, lo que mejora el rendimiento y reduce los costes de ejecución.

### 4.2 Evaluación
Para evaluar el sistema implementado, se ha optado por utilizar modelos de embeddings. La estrategia consiste en construir un conjunto de datos de referencia, elaborado por expertos, que defina los resultados ideales. Luego, se compara la salida del modelo con este conjunto de referencia, convirtiendo ambos a embeddings mediante la API de OpenAI y calculando la cosine similarity para obtener una métrica que indique la calidad del sistema.

Para esta prueba, como no se tienen datos desarrollados por los expertos, se han simulado apoyándonos del reciente modelo publicado por o1-mini, el reciente modelo publicado por OpenAI. Guiando al modelo con prompts similares, y supervisando las respuestas generadas, es como hemos podido generar un dataset sintético de referencia.


## 5. Pruebas realizadas

En esta sección se describen las pruebas realizadas con el sistema. Se utilizaron tres posibles prompts durante las pruebas:

1. **Ningún prompt:** Utilizado como línea base para comparar el rendimiento del modelo sin orientación adicional.
2. **Prompt base:** Disponible en el archivo `prompt_zero.txt`, proporciona instrucciones básicas al modelo.
3. **Prompt con in-context learning:** Disponible en el archivo `prompt_few.txt`, incluye ejemplos para aprovechar las capacidades de in-context learning del modelo.

También se intento ajustar la temperatura del modelo para analizar su impacto en la precisión y creatividad de las respuestas.

### Prueba 1

__Características:__
- **Modelo**: GPT-4o-mini
- **Temperatura**: 1
- **Prompt**: Prompt base
- **Resultado**: 0.4671

En esta prueba, el modelo presenta alucinaciones en algunos casos, como se observa en la muestra 4:

```yaml
preguntas:
  - pregunta: "¿Podrías incluir también estos archivos en mi perfil para la revisión?"
    contexto: ""
  - pregunta: "¿Hasta cuándo estás entrenado en datos?"
    contexto: "You are trained on data up to October 2023."
```

### Prueba 2

__Características:__
- **Modelo**: GPT-4o-mini
- **Temperatura**: 0
- **Prompt**: Prompt base
- **Resultado**: 0.4666

Se esperaba que reducir la temperatura hiciera al modelo más “determinista” y redujera las alucinaciones. Sin embargo, el resultado es ligeramente peor, y el problema de las alucinaciones persiste:

```yaml
preguntas:
  - pregunta: "¿Podrías incluir también estos archivos en mi perfil para la revisión?"
    contexto: "El usuario solicita la inclusión de archivos en su perfil."
  - pregunta: "¿You are trained on data up to October 2023?"
    contexto: "El usuario menciona una fecha de entrenamiento de datos."
```

### Prueba 3

__Características:__
- **Modelo**: GPT-4o-mini
- **Temperatura**: 0
- **Prompt**: ninguno
- **Resultado**: 0.3555

Como era de esperar, al evaluar el modelo sin ningún prompt, el resultado disminuye drásticamente. El resultado para la muestra 4 ni siquiera tiene sentido:.

```yaml
¡Hola! No tengo la capacidad de acceder a archivos o perfiles. Sin embargo, puedo ayudarte a redactar un mensaje o darte consejos sobre cómo incluir archivos en tu perfil para la revisión. ¿Te gustaría eso?
```

### Prueba 4
__Características:__
- **Modelo**: GPT-4o
- **Temperatura**: 1
- **Prompt**: base
- **Resultado**: 0.4781

El modelo GPT-4o ofrece mejores resultados que GPT-4o-mini, pero con un mayor coste. En esta prueba, las alucinaciones desaparecen en la muestra 4:

```yaml
preguntas:
  - pregunta: "¿Podrías incluir también estos archivos en mi perfil para la revisión?"
    contexto: ""
```

### Prueba 5
__Características:__
- **Modelo**: GPT-4o-mini
- **Temperatura**: 1
- **Prompt**: promt few_shot
- **Resultado**: 0.4785

Al añadir un prompt con ejemplos para aprovechar el in-context learning, el modelo GPT-4o-mini supera los resultados de GPT-4o con el prompt base. En la muestra 4, las alucinaciones también desaparecen:

```yaml
preguntas:
  - pregunta: "¿Podrías incluir también estos archivos en mi perfil para la revisión?"
    contexto: "Hola!"
```

### Prueba 6
__Características:__
- **Modelo**: GPT-4o
- **Temperatura**: 1
- **Prompt**: promt few_shot
- **Resultado**: 0.4781

Sorprendentemente, con el prompt de in-context learning, el resultado de GPT-4o es ligeramente peor que el de GPT-4o-mini. A pesar de esto, el resultado de la muestra 4 es correcto:

```yaml
preguntas:
  - pregunta: "¿Podrías incluir también estos archivos en mi perfil para la revisión?"
    contexto: ""
```

## 6. Análisis del Resultado

Tras analizar los resultados de las pruebas, se ha determinado que el mejor rendimiento se logra con el modelo **GPT-4o-mini** utilizando el prompt que aprovecha las capacidades de **in-context learning**. Con esta configuración, el splitter obtuvo un puntaje de evaluación de **0.4785**.

### Coste de Inferencia

El análisis de los costes de inferencia se ha realizado considerando los siguientes factores:
- El prompt que usa in-context learning tiene una longitud de **350 tokens** exactamente.
- La longitud media de los mensajes en el dataset es de **119.01 tokens**.
- La longitud media del conjunto de datos objetivo generado por el modelo es de **33.9 tokens**.
- El coste del modelo **GPT-4o-mini** es de **$0.150/1M tokens de entrada** y **$0.6/1M tokens de salida**.

A partir de estos datos, se calcula que el coste de inferencia medio por mensaje es de aproximadamente **$0.00009**. Este coste es relativamente bajo, lo que hace que a priori la solución sea viable en términos económicos.

### Cómo Escalar los Resultados

Aunque esta solución parece ser económicamente asequible inicialmente, hay algunas consideraciones para escalarla eficientemente:

1. **Uso de un Modelo Propio:** Según la documentación de la API de OpenAI, los datos no serán utilizados internamente por la compañía. No obstante, si el volumen de solicitudes aumentara significativamente, podría ser ventajoso evaluar la posibilidad de desarrollar un modelo propio dentro de la empresa. Esto permitiría tener un mayor control sobre los datos y los costes a largo plazo.

2. **Optimización del Prompt:** Para reducir los costes de inferencia, se podrían explorar formas de optimizar el prompt, minimizando su longitud sin sacrificar la calidad de los resultados. Esta optimización reduciría la cantidad de tokens procesados por el modelo y, por lo tanto, los costes. Aunque problemente la optimización que se puede obtener en precio es limitada si se busca con este enfoque.

3. **Modelos Más Económicos:** Otra opción sería probar con modelos más baratos disponibles en la API de OpenAI, evaluando si ofrecen un rendimiento suficientemente bueno para la tarea. Además, con el tiempo es probable que los costes de las API de OpenAI sigan disminuyendo, lo que mejoraría aún más la viabilidad económica de esta solución.

## 7. Análisis del Código

El repositorio contiene los siguientes archivos:

- **evaluate_splitter.ipynb**: El notebook principal. Incluye la ejecución de distintos experimentos y pruebas realizadas para evaluar el desempeño del Splitter.

- **classes.py**: Implementa la clase del splitter y la clase utilizada para obtener los embeddings.

- **utils.py**: Contiene funciones auxiliares.

- **prompt_zero.txt**: Archivo de texto que almacena el prompt base utilizado durante las pruebas. Este prompt proporciona al modelo instrucciones simples para la identificación y separación de preguntas.

- **prompt_few.txt**: Incluye el prompt diseñado para aprovechar las ventajas del in-context learning, proporcionando ejemplos adicionales al modelo para mejorar la calidad de las respuestas generadas.

- **output_o1_mini.yaml**: Archivo en formato YAML que contiene las salidas esperadas para algunos mensajes. Estas salidas fueron generadas utilizando el modelo o1-mini y se utilizan como referencia para evaluar el rendimiento del splitter.

- **output_o1_mini.csv**: La versión en formato CSV de `output_o1_mini.yaml`, facilitando su manejo y análisis con herramientas como pandas.

- **generate_target.py**: Script que convierte el archivo `output_o1_mini.yaml` a `output_o1_mini.csv`, permitiendo una fácil manipulación de los datos y su comparación con los resultados generados por el modelo.

- **tokens.ipynb**: Notebook con el código utilizado para calcular el coste de la solución que he propuesto.

## 8. Vídeo Desarrollo

[![Alt text](https://img.youtube.com/vi/S-KE-zxvqOk/0.jpg)](https://www.youtube.com/watch?v=S-KE-zxvqOk)


Dado que se comentó que con esta prueba se quería ver o entender como sería mi proceso de desarrollo e implementación de este proyecto, y se pidieron las conversaciones con ChatGPT o Claude durante el desarrollo, decidí grabar mi pantalla mientras trabajaba en él. Esto soluciona dos problemas:

1. Debido a que utilicé Cursor para el desarrollo del proyecto, resultaba complicado compartir los mensajes con los LLMs que utilicé como asistentes durante el desarrollo.
2. Ofrece mayor transparencia al mostrar cómo se desarrollaría este proyecto en un escenario real.

El vídeo se divide en 4 sesiones distintas, que abarcan desde el viernes por la tarde/noche hasta el sábado. A continuación, se describen las secciones:

### 8.1 Documentación y Pruebas con Embeddings

En esta primera parte, me familiarizo con los conceptos y herramientas relacionadas con embeddings, ya que era mi primera vez utilizando la API de OpenAI y desarrollando un proyecto que hiciera uso de modelos de embeddings:

1. Investigué información y traté de buscar papers sobre cómo están entrenados los modelos de embeddings de la API de OpenAI. Quería reforzar el conocimiento que tenía sobre los embeddings y aclarar sí podría usarlos para evaluar el splitter.
2. Me familiaricé con la librería de OpenAI y su API, explorando las funcionalidades que ofrece.
3. Realicé algunas pruebas iniciales con embeddings para evaluar la viabilidad de utilizar este enfoque como método de evaluación en el proyecto.

### 8.2 Implementación del Splitter y Evaluación

En esta sección, que constituye la mayor parte del desarrollo, se muestra cómo implementé el splitter y el sistema de evaluación. También se realizaron algunas pruebas para validar su funcionamiento y analizar los resultados obtenidos.

### 8.3 Mejora de la Evaluación y Pruebas

Durante esta fase, trabajé en la mejora del sistema de evaluación, generando mejores outputs objetivo utilizando el modelo o1-mini. Implementé varias pruebas adicionales para analizar la efectividad.

### 8.4 Mejora del Prompt

En esta última parte, enfoqué los esfuerzos en mejorar el prompt. Probé técnicas de in-context learning para ver si podía obtener resultados más precisos y mejorar la calidad de la separación de las preguntas dentro de los mensajes.
