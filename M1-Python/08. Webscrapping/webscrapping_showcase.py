import requests
from bs4  import BeautifulSoup

url = "https://www.goodreads.com/genres/historical-fiction"
'''
amazon = "https://www.amazon.es/?tag=bingamazoabk-21&hvadid=79852063932248&hvqmt=e&hvbmt=be&hvdev=c&ref=pd_sl_7opjaapkbf_e"

am_page = requests.get(amazon)

am_soup = BeautifulSoup(am_page.content, "html.parser")

print(am_page)
'''

page = requests.get(url)

soup = BeautifulSoup(page.content, "html.parser")

book_title = soup.find('a', class_ = "bookTitle")

book_link = book_title['href']


book_page = requests.get(book_link)

book_soup = BeautifulSoup(book_page.content, "html.parser")

title = book_soup.find('h1', class_ = "gr-h1 gr-h1--serif")


'''
div_content = soup.find_all('div', class_="giveawayPreviewBookContainer")


tittles = []
authors = []

second_div = div_content[0].find('div', class_="description descriptionContainer")

third_div = second_div.find('div', id = "bookAuthors")

spans = third_div.find_all("span")

print(spans[0].text)


for div in div_content:
    book_tittle = div.find('a', class_ = "bookTitle")
    tittles.append(book_tittle.text)
    author =  div.find('span', itemprop="name")
    authors.append(author.text)
print(tittles)
print(authors)


tittle = soup.find_all('h1', class_="left")

book_titles = soup.find_all('a', class_ = "bookTitle")
print(book_titles)

list_of_tittles = []
for book in book_titles:
    list_of_tittles.append(book.text)
'''


