import requests
import time
import pandas as pd
import os

class Crawler:
    def __init__(self, query="Artificial Intelligence", max_results=1500, save_path="./data/"):
        # Inisialisasi Crawler dengan query pencarian dan jumlah maksimum hasil.
        self.query = query
        self.max_results = max_results
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.params = {
            "query": self.query,
            "limit": 100,  # Ambil 100 data per batch
            "fields": "title,url,abstract,year,externalIds"
        }
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # Pastikan folder penyimpanan ada

    def crawl_semantic_scholar(self):
        # Melakukan crawling ke Semantic Scholar API.
        results = []
        next_cursor = None  # Untuk pagination

        while len(results) < self.max_results:
            try:
                if next_cursor:
                    self.params["next"] = next_cursor  # Gunakan cursor-based pagination
                
                response = requests.get(self.base_url, params=self.params, timeout=10)
                
                if response.status_code == 429:
                    print("Terlalu banyak permintaan! Menunggu 60 detik...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get("data"):  # Jika tidak ada hasil, hentikan
                    print("Tidak ada data tambahan, berhenti.")
                    break
                
                for paper in data["data"]:
                    results.append({
                        "title": paper.get("title", "Title Unavailable"),
                        "link": paper.get("url", "Link Unavailable"),
                        "doi": paper.get("externalIds", {}).get("DOI", "DOI Unavailable"),
                        "abstract": paper.get("abstract", "Abstract unavailable"),
                        "year": paper.get("year", "Unavailable")
                    })
                    
                    if len(results) >= self.max_results:
                        break
                
                # Ambil cursor untuk pagination
                next_cursor = data.get("next", None)
                if not next_cursor:
                    print("Tidak ada halaman berikutnya.")
                    break
                
                time.sleep(15)  # Hindari rate limit
                
            except requests.exceptions.RequestException as e:
                print(f"Terjadi kesalahan: {e}")
                time.sleep(240)  # Jika error, tunggu lama agar tidak diblokir
        
        return results

    def save_to_csv(self, data):
        # Menyimpan hasil crawling ke dalam file CSV.
        df = pd.DataFrame(data)
        file_path = os.path.join(self.save_path, "semantic_scholar_results.csv")
        df.to_csv(file_path, index=False)
        print(f"Data berhasil disimpan dalam {file_path}")
        return file_path

    def begin_crawl(self):
        # Memulai proses crawling dan menyimpan hasilnya.
        print(f"Memulai crawling untuk query: {self.query}")
        data = self.crawl_semantic_scholar()
        
        if not data:
            return "Crawling gagal atau tidak ada data yang ditemukan."
        
        return self.save_to_csv(data)
