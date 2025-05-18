import base64
from io import BytesIO
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from jinja2 import Template
import pdfkit
import os
from pathlib import Path
from collections import Counter

class ReportGenerator:
    def __init__(self):
        self.report_dir = 'reports'
        os.makedirs(self.report_dir, exist_ok=True)
        self.config = self._get_pdfkit_config()
        
    def _get_pdfkit_config(self):
        """Try to locate wkhtmltopdf executable"""
        possible_paths = [
            r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
            '/usr/local/bin/wkhtmltopdf',
            '/usr/bin/wkhtmltopdf'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return pdfkit.configuration(wkhtmltopdf=path)
        
        raise Exception("wkhtmltopdf not found. Please install it or check the path.")

    def _load_data(self):
        """Load processed data and metrics"""
        with open('data/processed/metrics.json') as f:
            metrics = json.load(f)
        
        df = pd.read_json('data/processed/semantic_scholar_results_clean.json')
        return metrics, df

    def _generate_visualizations(self, df):
        """Generate visualizations and return as base64 encoded string"""
        plt.figure(figsize=(12, 6))
        
        # Visualization 1: Publication year distribution
        if 'year' in df.columns:
            plt.subplot(1, 2, 1)
            df['year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
            plt.title('Publication Year Distribution', pad=20)
            plt.xlabel('Year')
            plt.ylabel('Count')
        
        # Visualization 2: Top processed words
        if 'processed_title' in df.columns:
            try:
                all_words = ' '.join(df['processed_title'].dropna()).split()
                word_freq = Counter(all_words).most_common(20)
                
                plt.subplot(1, 2, 2)
                pd.DataFrame(word_freq, columns=['Word', 'Count']).set_index('Word').plot(
                    kind='barh', color='lightgreen', legend=False
                )
                plt.title('Top 20 Processed Words', pad=20)
                plt.xlabel('Frequency')
            except Exception as e:
                print(f"Word frequency visualization error: {str(e)}")
        
        plt.tight_layout()
        
        # Save to in-memory buffer instead of file
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"

    def _generate_statistics(self, df):
        """Generate statistics section"""
        stats = []
        
        if 'citationCount' in df.columns:
            stats.append({
                'title': 'Citation Statistics',
                'items': [
                    f"Average citations: {df['citationCount'].mean():.1f}",
                    f"Max citations: {df['citationCount'].max()}",
                    f"Min citations: {df['citationCount'].min()}"
                ]
            })
        
        if 'influentialCitationCount' in df.columns:
            stats.append({
                'title': 'Influential Citations',
                'items': [
                    f"Average influential citations: {df['influentialCitationCount'].mean():.1f}"
                ]
            })
        
        return stats

    def _generate_html_template(self):
        """Return Jinja2 template for report"""
        return Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Semantic Scholar Dataset Report</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 1.5cm;
            font-size: 11pt;
            color: #333;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h2 { 
            color: #3498db; 
            margin-top: 25px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        h3 { 
            color: #2c3e50;
            margin-top: 20px;
        }
        .metrics { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px;
            margin-bottom: 20px;
        }
        img { 
            max-width: 100%; 
            height: auto; 
            margin: 15px auto;
            display: block;
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .warning { 
            color: #e74c3c; 
            background: #fdecea; 
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        .stat-block {
            margin: 15px 0;
            padding: 10px;
            background: #f5f9fc;
            border-left: 4px solid #3498db;
        }
        ul {
            margin-top: 5px;
            padding-left: 20px;
        }
        li {
            margin-bottom: 5px;
        }
        footer {
            margin-top: 40px;
            font-size: 0.8em; 
            color: #7f8c8d; 
            text-align: center;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }
    </style>
</head>
<body>
    {{ content }}
    <footer>
        Report generated automatically on {{ date }} using DVC pipeline
    </footer>
</body>
</html>
""")

    def generate_report(self):
      """Main method to generate the report"""
      try:
          # 1. Load data
          metrics, df = self._load_data()
          available_columns = df.columns.tolist()
          
          # Generate visualizations as base64
          viz_data = self._generate_visualizations(df)
          
          # 3. Generate statistics
          statistics = self._generate_statistics(df)
          
          # 4. Prepare report content
          report_content = f"""
  <h1>Technical Report: Semantic Scholar Dataset Analysis</h1>

  <div class="metrics">
      <p><strong>Report Generated</strong>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      
      <h2>Dataset Metadata</h2>
      <ul>
          <li>Original records: {metrics.get('original_count', 'N/A')}</li>
          <li>Processed records: {metrics.get('processed_count', 'N/A')}</li>
          <li>Processing date: {metrics.get('processing_date', 'N/A')}</li>
          <li>Available columns: {', '.join(available_columns)}</li>
      </ul>
  </div>

  <h2>Processing Steps</h2>
  <ol>
      <li>Text cleaning (lowercasing, punctuation removal)</li>
      <li>Stopword removal</li>
      <li>Lemmatization</li>
      <li>Title processing ('processed_title' column added)</li>
  </ol>

  <h2>Dataset Visualizations</h2>
  <img src="{viz_data}" alt="Dataset Visualizations" style="max-width: 100%; height: auto;">
  """

          # Add statistics section
          if statistics:
              report_content += "<h2>Statistical Analysis</h2>"
              for stat in statistics:
                  report_content += f"""
  <div class="stat-block">
      <h3>{stat['title']}</h3>
      <ul>
          {"".join(f"<li>{item}</li>" for item in stat['items'])}
      </ul>
  </div>
  """
          else:
              report_content += """
  <div class="warning">
      No citation statistics available in the dataset
  </div>
  """

          # 5. Generate PDF
          template = self._generate_html_template()
          pdf_path = os.path.join(self.report_dir, 'technical_report.pdf')
          
          pdfkit.from_string(
              template.render(
                  content=report_content,
                  date=datetime.now().strftime('%Y-%m-%d %H:%M')
              ),
              pdf_path,
              options={
                  'page-size': 'A4',
                  'margin-top': '15mm',
                  'margin-right': '15mm',
                  'margin-bottom': '15mm',
                  'margin-left': '15mm',
                  'encoding': 'UTF-8',
                  'quiet': ''
              },
              configuration=self.config
          )

          print(f"Technical report successfully generated: {pdf_path}")
          return True

      except Exception as e:
          print(f"Error generating report: {str(e)}")
          return False

if __name__ == "__main__":
    generator = ReportGenerator()
    generator.generate_report()