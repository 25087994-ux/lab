import requests
from bs4 import BeautifulSoup

URL = "https://movie.douban.com/top250"
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
page = requests.get(URL,headers=headers)
#print("\n***Page***")
#print(page.text)
#print("***END***")

soup = BeautifulSoup(page.content, "html.parser")
#print ("\n***Soup***")
#print(soup)
#print("***END***")

results = soup.find("div",id="content")
#print ("\n***results***")
#print(results.prettify())
#print("***END***")

print ("\n***movie elements***")
movie_elements = results.find_all("div", class_="item")
for movie_element in movie_elements:
    hd_element=movie_element.find("div",class_="hd")
    title_element = movie_element.find("span", class_="title")
    score_element = movie_element.find("span", class_="rating_num")
    review_element = movie_element.find("p", class_="quote")
    print(title_element)
    print(score_element)
    print(review_element)
    print()
    print(title_element.text)
    print(score_element.text)
    print(review_element.text)
    print()
    print(title_element.text.strip())
    print(score_element.text.strip())
    print(review_element.text.strip())
    print()
print("***END***")


#print(movies_element,9.3)
print ("\n***movie Element***")
movies = results.find_all("span", class_="rating_num",string=lambda text:text and"9.5"in text)
print("Number of elements: ", len(movies))
elements = [
    p_element.parent.parent.parent for p_element in movies
]
for movie_element in elements:
    title_element = movie_element.find("span", class_="title")
    score_element = movie_element.find("span", class_="rating_num")
    review_element = movie_element.find("p", class_="quote")
    print(title_element.text.strip())
    print(score_element.text.strip())
    print(review_element.text.strip())
    print()

# 1. 创建空列表，存放所有电影评分
scores_list = []

# 2. 遍历所有电影，提取评分并加入列表
for movie_element in movie_elements:
    title_element = movie_element.find("span", class_="title")
    score_element = movie_element.find("span", class_="rating_num")
    review_element = movie_element.find("p", class_="quote")

    print(title_element.text.strip())
    print(score_element.text.strip())
    print(review_element.text.strip())
    print()

    # 把评分转成数字，存入列表
    if score_element:
        score = float(score_element.text.strip())
        scores_list.append(score)

if scores_list:
    # 计算平均分
    avg = sum(scores_list) / len(scores_list)

    # 计算最高分（使用 max() 函数）
    highest_score = max(scores_list)

    print(" 平均分：", avg)
    print(" 最高分：", highest_score)
else:
    print("未获取到评分")
