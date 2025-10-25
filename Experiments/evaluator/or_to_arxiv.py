import requests
import re

def get_arxiv_id(openreview_id):
    """Get arXiv ID from OpenReview paper ID."""
    try:
        # Get OpenReview paper
        r = requests.get(f"https://api2.openreview.net/notes?id={openreview_id}")
        data = r.json()
        
        if not data.get('notes'):
            print("OpenReview paper not found")
            return "not found"
        
        title = data['notes'][0]['content']['title']
        if isinstance(title, dict):
            title = title.get('value', '')
        
        # Search arXiv
        query = requests.utils.quote(str(title))
        r = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=5")
        
        # Extract arXiv ID
        match = re.search(r'<id>http://arxiv\.org/abs/([^<]+)</id>', r.text)
        if not match:
            print(f"arXiv paper not found for title: {title}")
            print(f"Response preview: {r.text[:500]}")
            return "not found"
        return match.group(1)
    
    except Exception as e:
        print(f"Error: {e}")
        return "not found"

if __name__ == "__main__":
    print(get_arxiv_id("ZsP3YbYeE9"))
