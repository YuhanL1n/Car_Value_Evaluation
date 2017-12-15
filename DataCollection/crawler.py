"""
Author: Yijun Zhang
Time: 2017 Fall
About: Data Mining Project - data collection part

****************
The request header below is used on my own computer, please change if necessary.

Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8
Accept-Encoding:gzip, deflate, br
Accept-Language:en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4,zh-TW;q=0.2
Cache-Control:max-age=0
Connection:keep-alive
Cookie:cl_b=jFm7EPmi5xGSvSmFWvDnugTHf0E; cl_def_hp=boulder; cl_tocmode=sss%3Agrid
Host:boulder.craigslist.org
If-Modified-Since:Sun, 12 Nov 2017 23:19:37 GMT
Referer:https://boulder.craigslist.org/search/cta
Upgrade-Insecure-Requests:1
User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36
******************

"""

import requests
import csv
from lxml import html
import time


# make & model: 0, year: 1, price: 13
attr_dic = {
    'VIN: ': 2,
    'condition: ': 3,
    'cylinders: ': 4,
    'drive: ': 5,
    'fuel: ': 6,
    'paint color: ': 7,
    'odometer: ': 8,
    'size: ': 9,
    'title status: ': 10,
    'transmission: ': 11,
    'type: ': 12
}

csv_head = ['make and model', 'year', 'VIN', 'condition', 'cylinders', 'drive', 'fuel', 'color', 'odometer', 'size', 'title', 'transmission', 'type', 'price']


posts_per_city = dict()

header = {
    'Accept-Language': 'en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4,zh-TW;q=0.2',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection':  "keep-alive",
    'Pragma': 'No-cache',
    'Cache-Control': 'No-cache',
    'Upgrade-Insecure-Requests': '1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
}


def crawl_post(posturl, rows):
    r = requests.get(posturl, headers=header)
    p = r.content
    root = html.fromstring(p)

    attrType = [i.text for i in root.xpath("//p[@class='attrgroup']/span[not(@class = 'otherpostings')]")]
    attrValue = [i.text for i in root.xpath("//p[@class='attrgroup']/span[not(@class = 'otherpostings')]/b")]
    price = root.xpath("//span[@class='price']")[0].text[1:]

    car = attrValue[0].split(' ')
    year = car[0]
    make_model = ' '.join(car[1:])

    #  missing field
    row = ['None'] * 14
    row[0] = make_model
    row[1] = year
    row[13] = price

    for i in range(1, len(attrType)):
        row[attr_dic[attrType[i]]] = attrValue[i]

    rows.append(row)


def crawl_page(pageurl, host_name, rows):
    e_count = 0 #exception
    s_count = 0 #success
    r = requests.get(pageurl, headers=header)
    print("page response status: ", r.status_code)
    p = r.content
    root = html.fromstring(p)

    posts = root.xpath("//li[@class='result-row']/a[@href]/@href")
    next_url = host_name + root.xpath("//a[@class = 'button next']/@href")[0]

    print("scanning...")
    for url in posts:
        try:
            crawl_post(url, rows)
        except BaseException:
            e_count += 1
            pass
        else:
            s_count += 1

    print("good posts: ", s_count)
    print("bad posts: ", e_count)
    print()

    return next_url


def crawl(url, host_name, city):

    rows = [] # save all posts of a city
    print("city: ", city)
    r = requests.get(url, headers=header, allow_redirects=False)
    print("response status: ", r.status_code)
    p = r.content
    root = html.fromstring(p)
    total_count = root.xpath("//span[@class='totalcount']")[0].text

    # in this case, Craigslist offers a bug
    if total_count == 2500:
        page_num = int(int(total_count) / 120)
        posts_per_city[city] = 2400
    else:
        page_num = int(int(total_count) / 120) + 1
        posts_per_city[city] = total_count


    print("total posts:", total_count)
    print("total pages:", page_num)
    print()

    for i in range(0, page_num):
        print("page: ", i)
        try:
            url = crawl_page(url, host_name, rows)
        except BaseException:
            print("error occurs\n")
            pass

    file_name = city + '.csv'
    # change the location for different cities/states
    with open('CO/' + file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_head)
        writer.writerows(rows)
    f.close()


if __name__ == '__main__':

    # exception: Craigslist have another kind of url with domain name .craigslist.com.mx. Simply ignore this case
    # error: status HTTP 3XX. when the urls are different from what we expected, loop of redirection will cause error 
    
    # change the url for different cities
    # choose any city for each state and then cities nearby will be processed automatically
    
    first_url = "https://boulder.craigslist.org/search/cto"

    r = requests.get(first_url, headers=header)
    p = r.content
    root = html.fromstring(p)

    cities = root.xpath("//select[@id='areaAbb']/option/@value")
    print("list of cities nearby: ", cities)
    print()

    for city in cities:
        # .craigslist.com.mx
        header['Host'] = city + ".craigslist.org"
        # cars + trucks by owners
        host_name = "https://" + header['Host']
        url = host_name + "/search/cto"
        crawl(url, host_name, city)

    print(posts_per_city)

