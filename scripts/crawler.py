import requests
import time
import pandas as pd

# Fungsi untuk melakukan crawling dari Semantic Scholar API
def crawl_semantic_scholar(query, max_results=1500):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 100,  # Ambil 100 data per batch
        "fields": "title,url,abstract,year,externalIds"
    }
    
    results = []
    next_cursor = None  # Untuk pagination

    while len(results) < max_results:
        try:
            if next_cursor:
                params["next"] = next_cursor  # Gunakan cursor-based pagination
            
            response = requests.get(base_url, params=params, timeout=10)
            
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
                
                if len(results) >= max_results:
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

# Query pencarian
query = "Artificial Intelligence"
max_results = 1500

data = crawl_semantic_scholar(query, max_results)

df = pd.DataFrame(data)
df.to_csv('../data/semantic_scholar_results.csv', index=False)

print("Data berhasil disimpan dalam /data/semantic_scholar_results.csv")
