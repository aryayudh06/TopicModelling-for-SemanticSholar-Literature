from flask import Flask, jsonify, send_from_directory
import os
import sys
import logging
from flask_cors import CORS
from scripts.model_training import TopicModelTrainer

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Menambahkan path untuk mengakses modul di folder lain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import pipeline dari folder scripts
from scripts.crawler import Crawler
from scripts.preprocessing import Preprocessor
from scripts.visualization import Visualization

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Folder tempat menyimpan hasil charts
CHARTS_DIR = os.path.join(os.getcwd(), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)  # Buat folder jika belum ada

def run_crawl_pipeline():
    """Menjalankan pipeline crawling."""
    try:
        logging.info("Memulai proses crawling...")
        crawler = Crawler()
        crawl_result = crawler.begin_crawl()
        logging.info("Crawling selesai.")
        return crawl_result
    except Exception as e:
        logging.error(f"Error saat menjalankan crawling: {str(e)}")
        return {"error": str(e)}

def run_preprocess_pipeline():
    """Menjalankan pipeline preprocessing."""
    try:
        logging.info("Memulai preprocessing...")
        preprocessor = Preprocessor()
        processed_data_path = preprocessor.run()
        logging.info("Preprocessing selesai.")
        return processed_data_path
    except Exception as e:
        logging.error(f"Error saat menjalankan preprocessing: {str(e)}")
        return {"error": str(e)}

def run_visualization_pipeline():
    """Menjalankan pipeline visualisasi."""
    try:
        logging.info("Memulai visualisasi...")
        visualizer = Visualization()
        visualizer.generate_charts()
        logging.info("Visualisasi selesai.")
    except Exception as e:
        logging.error(f"Error saat menjalankan visualisasi: {str(e)}")
        return {"error": str(e)}
    
def run_training_pipeline():
    """Menjalankan pipeline training topic modeling."""
    try:
        logging.info("Memulai training model topik...")
        trainer = TopicModelTrainer()
        result = trainer.run()
        logging.info("Training selesai.")
        return result
    except Exception as e:
        logging.error(f"Error saat training: {str(e)}")
        return {"error": str(e)}


@app.route('/run-crawl', methods=['GET'])
def run_crawl():
    """Endpoint untuk menjalankan pipeline crawling."""
    result = run_crawl_pipeline()
    return jsonify({
        "message": "Pipeline crawling selesai dijalankan.",
        "result": result
    })

@app.route('/run-preprocess', methods=['GET'])
def run_preprocess():
    """Endpoint untuk menjalankan pipeline preprocessing."""
    result = run_preprocess_pipeline()
    return jsonify({
        "message": "Pipeline preprocessing selesai dijalankan.",
        "result": result
    })

@app.route('/run-visualize', methods=['GET'])
def run_visualize():
    """Endpoint untuk menjalankan pipeline visualisasi."""
    result = run_visualization_pipeline()
    
    # Pastikan hanya file gambar yang dikembalikan
    allowed_extensions = {".png", ".jpg", ".jpeg", ".svg"}
    charts = [
        f"/charts/{file}" for file in os.listdir(CHARTS_DIR) 
        if os.path.splitext(file)[1].lower() in allowed_extensions
    ]

    return jsonify({
        "message": "Pipeline visualisasi selesai dijalankan.",
        "charts": charts
    })

@app.route('/run-training', methods=['GET'])
def run_training():
    """Endpoint untuk menjalankan training model topik."""
    result = run_training_pipeline()
    return jsonify({
        "message": "Training selesai dijalankan.",
        "result": result
    })

@app.route('/charts/<filename>', methods=['GET'])
def get_chart(filename):
    """Endpoint untuk mengambil file chart yang sudah dibuat."""
    try:
        return send_from_directory(CHARTS_DIR, filename)
    except Exception as e:
        logging.error(f"Error saat mengakses file {filename}: {str(e)}")
        return jsonify({"error": f"Tidak dapat mengakses file {filename}"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
