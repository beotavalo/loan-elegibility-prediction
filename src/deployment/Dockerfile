# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt e instalar las dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY /app/ /app/

# Exponer el puerto 5000 para Gunicorn
EXPOSE 5000

# Comando para ejecutar Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
