# Generador de Texto con BLOOM

## Descripción

Este proyecto utiliza el modelo de lenguaje **BLOOM-560M** de **Hugging Face** para generar texto de forma interactiva, con una interfaz gráfica desarrollada en **Tkinter**. El modelo de lenguaje se basa en una arquitectura de transformación, lo que le permite generar secuencias de texto a partir de un texto inicial proporcionado por el usuario. Los parámetros de la generación son configurables para controlar la longitud, creatividad y otros aspectos de los resultados.

El generador permite que los usuarios ingresen un texto inicial, seleccionen algunas configuraciones de generación y vean los resultados generados en tiempo real.

## Características

- **Generación de texto**: Basado en el modelo **BLOOM-560M**, una red neuronal entrenada para generar texto.
- **Interfaz Gráfica**: Desarrollada con **Tkinter** para una experiencia de usuario fluida y sencilla.
- **Configuración personalizada**: Permite al usuario ajustar parámetros como la longitud del texto, la creatividad, el número de resultados y más.
- **Salida visual**: Los textos generados se muestran en un área de texto desplazable que permite ver múltiples resultados.
- **Fácil de usar**: Simplemente ingresa un texto inicial y haz clic en "Generar Texto" para obtener las continuaciones.

## Requisitos

Antes de ejecutar este proyecto, asegúrate de tener instaladas las dependencias necesarias. Los paquetes requeridos son:

- **Tkinter** (incluido en la instalación estándar de Python, pero en algunos sistemas es necesario instalarlo).
- **Transformers**: Biblioteca para la integración con modelos de lenguaje de Hugging Face.

Puedes instalar las dependencias ejecutando el siguiente comando:

```
pip install transformers
```

## Parámetros del Modelo

- **Max Length**: Longitud máxima del texto generado.
- **Temperature**: Controla la creatividad del modelo. A temperaturas más altas (por ejemplo, 1.0), el modelo es más aleatorio. A temperaturas más bajas (por ejemplo, 0.2), las respuestas son más conservadoras.
- **Top K**: Determina cuántas palabras o tokens se consideran durante la generación. Valores más bajos hacen que la salida sea más coherente, pero menos creativa.
- **Num Return Sequences**: Cuántas secuencias de texto se generan.

## Flujo de la aplicación

### Paso 1: Ingresar el Texto Inicial

El usuario debe ingresar un texto inicial en el campo correspondiente en la interfaz gráfica.

### Paso 2: Configurar los Parámetros

El usuario puede ajustar los parámetros del modelo según sus preferencias, como la longitud del texto, la creatividad, y el número de secuencias a generar.

### Paso 3: Generar Texto

Haz clic en el botón **Generar Texto** para que el modelo procese el texto inicial y genere las continuaciones. Los resultados aparecerán en el área de texto de la parte inferior.

### Paso 4: Salir

Si deseas cerrar la aplicación, simplemente escribe **salir** en el campo de entrada y presiona el botón **Generar Texto**. La aplicación se cerrará automáticamente.

## Código

El archivo principal de este proyecto es `generador_texto.py`, que contiene el código necesario para cargar el modelo de lenguaje, generar texto y manejar la interfaz gráfica. A continuación, se describe el flujo del código:

### Carga del Modelo

El modelo y el tokenizador de **BLOOM** se cargan utilizando la librería **transformers** de Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
```

### Función de Generación de Texto

La función `generar_texto` toma un texto inicial y otros parámetros como la longitud máxima y el número de secuencias de texto a generar. Esta función procesa el texto con el modelo y devuelve las secuencias generadas:

```python
def generar_texto(texto_inicial, max_length=100, temperature=0.7, top_k=50, num_return_sequences=1):
    """
    Genera texto utilizando el modelo BLOOM.
    
    :param texto_inicial: Texto inicial para la generación.        
    :param max_length: Longitud máxima del texto generado.
    :param temperature: Control de la creatividad en la generación de texto.
    :param top_k: Número de palabras consideradas durante la generación.
    :param num_return_sequences: Número de secuencias generadas.
    :return: Lista de textos generados.
    """
    inputs = tokenizer(
        texto_inicial, 
        return_tensors="pt", 
        max_length=max_length,
        truncation=True, 
        padding=True,
        return_attention_mask=True
    )
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    textos_generados = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return textos_generados
```

### Interfaz Gráfica

La interfaz gráfica está construida utilizando **Tkinter**. Se define una ventana principal con un campo de entrada para el texto inicial, un botón para generar el texto y un área de texto desplazable para mostrar los resultados:

```python
import tkinter as tk
from tkinter import scrolledtext

root = tk.Tk()
root.title("Generador de Texto con BLOOM")          
root.geometry("700x500")
root.config(bg="#212121")

style = {
    'bg': '#212121',  
    'fg': '#E0E0E0', 
    'button_bg': '#3F51B5', 
    'button_fg': '#FFFFFF',  
    'entry_bg': '#333333', 
    'entry_fg': '#E0E0E0',  
    'text_bg': '#1A1A1A',  
    'text_fg': '#E0E0E0',  
    'border_color': '#888888',
}

entry_label = tk.Label(root, text="Ingresa el texto inicial:", fg=style['fg'], bg=style['bg'], font=("Arial", 14, "bold"))
entry_label.pack(pady=20)

entry = tk.Entry(root, width=60, font=("Arial", 14), bg=style['entry_bg'], fg=style['entry_fg'], borderwidth=2, relief="solid", bd=1)
entry.pack(pady=10)

send_button = tk.Button(root, text="Generar Texto", command=on_enviar, font=("Arial", 14, "bold"), bg=style['button_bg'], fg=style['button_fg'], relief="flat")
send_button.pack(pady=20)

output_text = scrolledtext.ScrolledText(root, width=70, height=15, font=("Arial", 12), bg=style['text_bg'], fg=style['text_fg'], wrap=tk.WORD, bd=1, relief="solid")
output_text.pack(pady=20)

root.mainloop()
```

## Contribución

Si deseas mejorar el proyecto, puedes hacerlo a través de un fork y enviar un pull request. Algunas ideas para mejoras incluyen:

- Añadir más modelos de lenguaje o soporte para otros idiomas.
- Mejorar la interfaz gráfica con más opciones de personalización.
- Agregar funcionalidades adicionales como la posibilidad de guardar el texto generado.

## Créditos

- **Hugging Face**: El modelo BLOOM y la librería transformers.
- **Tkinter**: Para la creación de la interfaz gráfica. 
- **GitHub**: TakizawaXD para el repositorio de este proyecto.

## Diagrama de Flujo

```plaintext
+----------------------------+
|   Inicio del Programa      |
+----------------------------+
              |
              v
+----------------------------+
|   El usuario ingresa texto |
+----------------------------+
              |
              v
+----------------------------+
|   El texto es procesado    |
|    por el modelo BLOOM     |
+----------------------------+
              |
              v
+----------------------------+
|    Mostrar el resultado    |
|   (Texto generado)         |
+----------------------------+
              |
              v
+----------------------------+
|   ¿Usuario escribe 'salir'?|
+----------------------------+
              |
      +-------+-------+
      |               |
      v               v
+------------------+    +----------------------------+
|  Fin del Programa|    | Continúa esperando nuevo  |
|  (mensaje final) |    | texto                      |
+------------------+    +----------------------------+
```

---

# **Analizador de Sentimientos con Transformers**

## **Descripción**

Este proyecto es un analizador de sentimientos basado en el modelo de lenguaje de **Hugging Face**, que utiliza el pipeline de `sentiment-analysis` para clasificar un texto como **POSITIVO** o **NEGATIVO**. El programa interactúa con el usuario en la consola, donde puede ingresar cualquier texto y recibir el resultado del análisis junto con un nivel de confianza.

## **Funcionalidades**

- **Análisis de Sentimientos**: El programa utiliza el modelo de Hugging Face para clasificar textos como positivos o negativos.
- **Interactividad**: El usuario puede ingresar textos de manera repetida y obtener los resultados en tiempo real.
- **Salida**: El análisis se realiza en la consola, y

 el resultado incluye la clasificación de sentimiento y el nivel de confianza.
- **Opción de Salir**: El usuario puede escribir `salir` para terminar el análisis y cerrar el programa.

## **Requisitos**

Este proyecto utiliza Python y la librería `transformers` de Hugging Face. Asegúrate de tener Python 3.6 o superior instalado.

### **Instalación de dependencias**

1. **Instalar Python**: Si no tienes Python instalado, puedes descargarlo desde [aquí](https://www.python.org/downloads/).
2. **Instalar las dependencias**:

   Abre tu terminal o consola y ejecuta el siguiente comando para instalar la librería `transformers`:

   ```bash
   pip install transformers
   ```

3. Instalar **torch** (si no está instalado automáticamente):

   ```bash
   pip install torch
   ```

### **Ejecutar el script**:

Guarda el código en un archivo llamado `analizador_sentimientos.py`. Para ejecutar el script, usa el siguiente comando:

```bash
python analizador_sentimientos.py
```

## **Ejemplo de ejecución**

```bash
¡Bienvenido al analizador de sentimientos! 😊
Escribe un texto y te diré si el sentimiento es POSITIVO o NEGATIVO.
Para salir, escribe 'salir'.

Ingresa un texto: Me siento muy feliz hoy
Resultado: POSITIVO (confianza: 0.99)

Ingresa un texto: Estoy muy triste
Resultado: NEGATIVO (confianza: 0.95)

Ingresa un texto: salir
¡Gracias por usar el analizador de sentimientos! 🌟
```

### Diagrama de flujo

```plaintext
+----------------------------+
|   Inicio del Programa      |
+----------------------------+
              |
              v
+----------------------------+
|   El usuario ingresa texto |
+----------------------------+
              |
              v
+----------------------------+
|   El texto es analizado    |
|    por la API de           |
|    Sentimientos (simulado) |
+----------------------------+
              |
              v
+----------------------------+
|    Resultado analizado     |
|   (Sentimiento + Confianza)|
+----------------------------+
              |
              v
+----------------------------+
|    Muestra el resultado    |
|   en la interfaz           |
+----------------------------+
              |
              v
+----------------------------+
|   ¿Usuario escribe 'salir'?|
+----------------------------+
              |
      +-------+-------+
      |               |
      v               v
+------------------+    +----------------------------+
|  Fin del Programa|    | Continúa esperando nuevo  |
|  (mensaje final) |    | texto                      |
+------------------+    +----------------------------+
```




Proyecto 1 - Generador de Texto con BLOOM

https://github.com/TakizawaXD/ChatBot



proyecto 2 - Analizador de Sentimientos con Transformers

https://psicoloia.netlify.app/

Proyecto 3 - IA DE IMAGENES CON TRANSFORMERS

https://huggingface.co/spaces/AndresTaker/IATAKER/blob/main/README.md?code=true
