import requests
import os

def search_github(query):
    print(f"Searching GitHub for: {query}")
    url = f"https://api.github.com/search/repositories?q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        items = response.json().get('items', [])
        for item in items[:5]:
            print(f"- {item['full_name']}: {item['html_url']}")
            print(f"  Description: {item['description']}")
    else:
        print(f"Failed to search GitHub: {response.status_code}")

search_github("RecGRELA")
search_github("PANTHER sequential user behavior")
search_github("attention economy simulation")
search_github("linear attention sequential recommendation")

