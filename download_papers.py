import arxiv
import os

def download_by_title(title, filename):
    client = arxiv.Client()
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=5
    )
    
    results = list(client.results(search))
    for paper in results:
        if title.lower() in paper.title.lower():
            print(f"Downloading {paper.title} to {filename}")
            paper.download_pdf(dirpath="papers", filename=filename)
            return True
    return False

os.makedirs("papers", exist_ok=True)
download_by_title("Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation", "rec_grela.pdf")
