import os
import time
import random
import logging
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_vestnik_archive(save_dir: str = r"E:\VKR\data\raw_pdf") -> None:
    os.makedirs(save_dir, exist_ok=True)

    base_url = "http://www.vestnik.vsu.ru/content/analiz/"
    archive_url = "http://www.vestnik.vsu.ru/content/analiz/archive_ru.asp"
    headers = {"User-Agent": "Mozilla/5.0"}

    logging.info(f"Target directory: {save_dir}")

    try:
        response = requests.get(archive_url, headers=headers, timeout=20)
        
        if response.status_code != 200:
            archive_url = "http://www.vestnik.vsu.ru/content/analiz/Archive_ru.asp"
            response = requests.get(archive_url, headers=headers, timeout=20)

        response.encoding = 'windows-1251'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        toc_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text()
            parent_text = a.parent.get_text() if a.parent else ""
            
            for year in range(2010, 2020):
                if str(year) in text or str(year) in parent_text:
                    if 'pdf' not in href.lower():
                        toc_links.append((year, urljoin(base_url, href)))
        
        toc_links = list(set(toc_links))
        logging.info(f"Index pages found: {len(toc_links)}")

        if not toc_links:
            return

        for year, url in sorted(toc_links, reverse=True):
            logging.info(f"Processing {year}: {url}")
            try:
                res = requests.get(url, headers=headers, timeout=20)
                res.encoding = 'windows-1251'
                issue_soup = BeautifulSoup(res.text, 'html.parser')
                
                downloaded_count = 0
                for el in issue_soup.find_all('a', href=True):
                    if '.pdf' in el['href'].lower():
                        pdf_url = urljoin(url, el['href'])
                        file_path = os.path.join(save_dir, f"{year}_{pdf_url.split('/')[-1]}")
                        
                        if not os.path.exists(file_path):
                            pdf_res = requests.get(pdf_url, headers=headers, timeout=60)
                            with open(file_path, 'wb') as f:
                                f.write(pdf_res.content)
                            downloaded_count += 1
                            time.sleep(random.uniform(0.5, 1.5))
                
                logging.info(f"Downloaded files for {year}: {downloaded_count}")
                
            except Exception as e:
                logging.error(f"Page processing error {url}: {e}")

    except Exception as e:
        logging.error(f"Network error: {e}")

if __name__ == "__main__":
    download_vestnik_archive()