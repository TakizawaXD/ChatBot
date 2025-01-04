import warnings
import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoTokenizer, AutoModelForCausalLM


warnings.filterwarnings("ignore", category=UserWarning)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

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

def on_enviar():
    """
    Función que se ejecuta cuando el usuario hace clic en el botón de generar texto.
    """
    texto_inicial = entry.get()
    if texto_inicial.lower() == 'salir':
        root.quit()
    else:
        textos = generar_texto(
            texto_inicial=texto_inicial,
            max_length=100,
            temperature=0.7,
            top_k=50,
            num_return_sequences=1
        )
        output_text.delete(1.0, tk.END)
        for idx, texto in enumerate(textos, start=1):
            output_text.insert(tk.END, f"{idx}. {texto}\n\n")
entry_label = tk.Label(root, text="Ingresa el texto inicial:", fg=style['fg'], bg=style['bg'], font=("Arial", 14, "bold"))
entry_label.pack(pady=20)

entry = tk.Entry(root, width=60, font=("Arial", 14), bg=style['entry_bg'], fg=style['entry_fg'], borderwidth=2, relief="solid", bd=1)
entry.pack(pady=10)

send_button = tk.Button(root, text="Generar Texto", command=on_enviar, font=("Arial", 14, "bold"), bg=style['button_bg'], fg=style['button_fg'], relief="flat")
send_button.pack(pady=20)

output_text = scrolledtext.ScrolledText(root, width=70, height=15, font=("Arial", 12), bg=style['text_bg'], fg=style['text_fg'], wrap=tk.WORD, bd=1, relief="solid")
output_text.pack(pady=20)

root.mainloop()
