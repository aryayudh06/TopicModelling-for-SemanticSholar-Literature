# Gunakan image Python sebagai base
FROM python:3.10.6

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Buka port yang akan digunakan
EXPOSE 5000

RUN ls -R /app

# Jalankan aplikasi Flask
CMD ["python", "/app/services/server/server.py"]
