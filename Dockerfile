# Gunakan image Python sebagai base
FROM python:3.10.6

# Set direktori kerja di dalam container
WORKDIR /app

# Salin hanya requirement.txt dulu untuk caching
COPY requirement.txt .

# Install dependencies lebih awal
RUN pip install --no-cache-dir -r requirement.txt

# Baru salin semua kode setelah itu
COPY . .

# Buka port yang akan digunakan
EXPOSE 5000

# (Opsional) Lihat isi folder saat build untuk debugging
RUN ls -R /app

# Jalankan aplikasi Flask
CMD ["python", "/app/services/server/server.py"]
