import os
import csv
from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler

# custom downloader to log source urls while scraping
class UrlLoggerDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        filename = super().get_filename(task, default_ext)
        group = getattr(self, 'current_group', '')
        query = getattr(self, 'current_query', '')
        
        if hasattr(self, 'csv_writer'):
            self.csv_writer.writerow([filename, task['file_url'], group, query])
            self.csv_file.flush()
        return filename

def scrape_images(queries, output_dir, label_type, max_num=60):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "raw_urls.csv")
    
    csv_file = open(csv_path, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if os.stat(csv_path).st_size == 0:
        csv_writer.writerow(["temp_filename", "source_url", "group", "query"])

    # save each query in a subfolder to avoid filename collisions
    for idx, q in enumerate(queries):
        sub_dir = os.path.join(output_dir, f"batch_{idx+1}")
        os.makedirs(sub_dir, exist_ok=True)
        print(f"[{label_type}] downloading: {q}")
        
        crawler = BingImageCrawler(
            downloader_cls=UrlLoggerDownloader,
            downloader_threads=4,
            storage={'root_dir': sub_dir}
        )
        crawler.downloader.csv_writer = csv_writer
        crawler.downloader.csv_file = csv_file
        crawler.downloader.current_group = f"batch_{idx+1}"
        crawler.downloader.current_query = q
        
        crawler.crawl(keyword=q, max_num=max_num)

    csv_file.close()
    print(f"done: {label_type} urls saved to {csv_path}")

if __name__ == "__main__":
    # positive queries: different angles, lighting, scales
    pos_queries = [
        "Ford car front grille emblem",
        "Ford car rear tailgate badge",
        "Ford dealership sign night",
        "Ford steering wheel logo closeup",
        "Ford key fob logo",
        "Ford F-150 front emblem",
        "Ford Mustang horse and ford badge",
        "vintage Ford car badge"
    ]
    
    # negative queries: hard negatives (blue ovals, other car brands, debadged)
    neg_queries = [
        "Subaru front grille emblem",
        "Hyundai front grille logo",
        "Toyota front emblem closeup",
        "Kia new logo car front",
        "custom debadged car front grille",
        "blue oval company logos"
    ]

    scrape_images(pos_queries, "raw_data/positive", "pos", max_num=65)
    scrape_images(neg_queries, "raw_data/negative", "neg", max_num=45)