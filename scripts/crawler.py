import requests
import time
import pandas as pd
import os
import json

class Crawler:
    def __init__(self, query="Artificial Intelligence", max_results=1500, save_path="./data/"):
        self.query = query
        self.max_results = max_results
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.params = {
            "query": self.query,
            "limit": 100,
            "fields": "title,url,abstract,year,externalIds,authors,journal,venue"
        }
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def crawl_semantic_scholar(self):
        results = []
        next_cursor = None

        while len(results) < self.max_results:
            try:
                if next_cursor:
                    self.params["next"] = next_cursor
                
                response = requests.get(self.base_url, params=self.params, timeout=10)
                
                if response.status_code == 429:
                    print("Rate limit exceeded! Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data.get("data"):
                    print("No more data available, stopping.")
                    break
                
                for paper in data["data"]:
                    authors_list = [author["name"] for author in paper.get("authors", [])] if paper.get("authors") else []
                    
                    results.append({
                        "title": paper.get("title", "Title Unavailable"),
                        "abstract": paper.get("abstract", "Abstract unavailable"),
                        "authors": authors_list,
                        "journal_conference_name": paper.get("journal", {}).get("name", "") 
                                            if paper.get("journal") 
                                            else paper.get("venue", "Unavailable"),
                        "publisher": paper.get("journal", {}).get("publisher", "Unavailable") 
                                if paper.get("journal") 
                                else "Unavailable",
                        "year": paper.get("year", "Unavailable"),
                        "doi": paper.get("externalIds", {}).get("DOI", "DOI Unavailable"),
                        "group_name": self.query
                    })
                    
                    if len(results) >= self.max_results:
                        break
                
                next_cursor = data.get("next", None)
                if not next_cursor:
                    print("No more pages available.")
                    break
                
                time.sleep(15)
                
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}")
                time.sleep(240)
        
        return results

    def save_to_csv(self, data):
        df = pd.DataFrame(data)
        # Handle list-type columns for CSV
        df['authors'] = df['authors'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        
        csv_path = os.path.join(self.save_path, "semantic_scholar_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Data successfully saved to CSV: {csv_path}")
        return csv_path

    def save_to_json(self, data):
        json_path = os.path.join(self.save_path, "semantic_scholar_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to JSON: {json_path}")
        return json_path

    def begin_crawl(self):
        print(f"Starting crawl for query: {self.query}")
        data = self.crawl_semantic_scholar()
        
        if not data:
            print("Crawling failed or no data found.")
            return None, None
        
        csv_file = self.save_to_csv(data)
        json_file = self.save_to_json(data)
        
        return csv_file, json_file