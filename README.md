# Modelo de Clasificación de Texto
Este proyecto consiste en un modelo de clasificación de texto utilizando modelos de procesamiento de lenguaje natural de HuggingFace.

## Requisitos
- Python 3.7 o superior
- `transformers` de HuggingFace
- `datasets` de HuggingFace
- Jupyter Notebook

## Instalación
Para instalar las dependencias necesarias, puedes utilizar el siguiente comando:
pip install transformers datasets


## Uso
Carga del Dataset
El dataset debe ser tokenizado antes de ser utilizado para entrenar el modelo. En este caso, se utiliza un tokenizador específico de HuggingFace.

## Preprocesamiento del Dataset
Es necesario dividir el dataset en un conjunto de entrenamiento y uno de prueba. Esto incluye revolver el dataset y particionarlo en 80% para entrenamiento y 20% para prueba.

# Revolver el dataset
new_tokenized_dataset = tokenized_dataset["train"].shuffle()

# Calcular el número de elementos del dataset
len_dataset = len(tokenized_dataset["train"])

# Partir el dataset en dos trozos
train_dataset = tokenized_dataset["train"][:int(len_dataset * 0.8)]
test_dataset = tokenized_dataset["train"][int(len_dataset * 0.8):]

# Construir el nuevo dataset
new_dataset = DatasetDict({
    "train": Dataset.from_dict(train_dataset),
    "test": Dataset.from_dict(test_dataset)
})

## Definición y Entrenamiento del Modelo
Definir el modelo de clasificación y entrenarlo con el conjunto de datos tokenizados.

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained("nombre_del_modelo")
model = AutoModelForSequenceClassification.from_pretrained("nombre_del_modelo", num_labels=2)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=new_dataset["train"],
    eval_dataset=new_dataset["test"],
)

# Entrenar el modelo
trainer.train()

## Evaluación del Modelo
Evaluar la precisión del modelo utilizando el conjunto de prueba.
trainer.evaluate()

## Uso del Modelo
Utilizar el modelo entrenado para clasificar nuevos textos.
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ejemplos de clasificación
print(classifier("Texto de ejemplo"))
